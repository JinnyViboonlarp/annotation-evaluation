import sys
import os
import re
import spacy

# this method find 'bad indices' (i.e. before the first ':' of each line in the transcript)
def find_bad_indices(filepath):
    infile = open(filepath, 'r')
    bad_list = []
    lines = infile.readlines()
    first_i = 0
    for line in lines:
        last_i = (first_i + len(line.split(':')[0]))
        if(':' in line):
            bad_list.append((line.split(':')[0], first_i, (last_i+1)))
        first_i = (first_i + len(line))
    return bad_list

# this method remove entities in the 'bad indices' range (i.e. before the first ':' of each line in the transcript) \
# from the entity_list
def remove_bad_indices(bad_list, entity_list):
    # compare the two list in the way similar to mergesort
    i1 = 0; i2 = 0
    while((i1 < len(entity_list)) and (i2 < len(bad_list))):
        while((i1 < len(entity_list)) and (entity_list[i1][1] < bad_list[i2][2])): # [1] = start index of the entity, [2] = end index
            if(entity_list[i1][2] > bad_list[i2][1]): # the entity pointed by i1 overlaps with the forbidden indices
                entity_list.pop(i1)
            else:
                i1 += 1
        i2 += 1
    return entity_list

def clean_all_transcripts():
    # this function read transcripts files and clean them of 'bad indices'
    before_clean_dir = "transcripts"
    after_clean_dir = "transcripts-cleaned"
    for text_name in os.listdir(before_clean_dir):
        if(text_name.endswith(".txt")):
            inpath = before_clean_dir + '/' + text_name
            bad_list = find_bad_indices(inpath)
            infile = open(inpath, 'r')
            text = infile.read()
            infile.close()
            text_list = list(text) # python string is immutable, but we want to modify input_text
            for ( _, start_i, end_i) in reversed(bad_list):
                if((end_i < len(text)) and text[end_i] == ' '):
                    text_list.pop(end_i)
                for i in range(end_i-1, start_i-1, -1): # the range is [start_i, end_i)
                    text_list.pop(i)
            text = "".join(text_list)
            outpath = after_clean_dir + '/' + text_name
            outfile = open(outpath, 'w')
            outfile.write(text)
            outfile.close()

def truecase(text):
    # this function is currently only used by dbpedia model.
    ner = spacy.load("ner_models/model-best-uncased-sm")
    doc = ner(text)

    tok_idx = {}
    for (n, tok) in enumerate(doc):
        p1 = tok.idx
        p2 = p1 + len(tok.text)
        tok_idx[n] = (p1, p2)

    interested_labels = ('ORG','NORP','PERSON','GPE','LOC','FAC','PRODUCT','EVENT','WORK_OF_ART') # must include 'PER' if trained with conll terminology
    text_list = list(text) # python string is immutable, but we want to modify input_text
    text = text.replace('.',' ') # a trick to make re.finditer recognizes dot. The real text is not affected since it is stored in text_list
    text = text.replace('-',' ') # again, a trick to make re.finditer recognizes dash.
    for (n, ent) in enumerate(doc.ents):
        if(ent.label_ in interested_labels):
            start = tok_idx[ent.start][0]; end = tok_idx[ent.end - 1][1]
            entity_text = text[start:end]
            for m in re.finditer(r'\S+', entity_text):
                capitalized_index = (start + m.start())
                text_list[capitalized_index] = text_list[capitalized_index].upper()
    text = "".join(text_list)
    return text

def write_entities(text, entity_list, outpath):
    # write entities from entity_list to output file 
    outfile = open(outpath, 'w')
    for i, [label, start, end] in enumerate(entity_list):
        outfile.write(('T'+str(i+1)) + '\t' + label + ' ' + str(start) + ' ' + str(end) + '\t' + repr(text[start:end])[1:-1] + '\n')
        # repr(ent.text)[1:-1] prints newline character as the substring '\n'
    outfile.close()    

def annotate(filepath, choice='default'):
    infile = open(filepath, 'r')
    text = infile.read()
    infile.close()

    if(choice=='default'):
        nlp = spacy.load("en_core_web_sm") # TODO: specify the model to do only NER task, to save time
    elif(choice=='uncased'):
        text = text.lower()
        nlp = spacy.load("ner_models/model-best-uncased-sm")
    elif(choice=='force-uncased'): # uncased data but default model
        text = text.lower()
        nlp = spacy.load("en_core_web_sm")
    else:
        print("ERROR: parameter 'choice' has invalid value")
        sys.exit()
        #doc = nlp(truecase(text.lower()))

    doc = nlp(text)
    #print('detecting named entities..')
    #print('annotating..')

    tok_idx = {}
    for (n, tok) in enumerate(doc):
        p1 = tok.idx
        p2 = p1 + len(tok.text)
        tok_idx[n] = (p1, p2)

    interested_labels = ('ORG','NORP','PERSON','GPE','LOC','FAC','PRODUCT','EVENT','WORK_OF_ART') # must include 'PER' if trained with conll terminology
    entity_list = [[ent.label_, tok_idx[ent.start][0], tok_idx[ent.end - 1][1]] for ent in doc.ents if ent.label_ in interested_labels]
    # each sublist in entity_list is [label, start, end]
    entity_list.sort(key=lambda x: x[2]) # sorted by end character's index
    bad_list = find_bad_indices(filepath) # find 'bad indices' (i.e. before the first ':' of each line in the transcript)
    entity_list = remove_bad_indices(bad_list, entity_list) # remove entities in the 'bad indices' ranges

    # write entities to output file 
    filename = os.path.basename(filepath)
    outname = choice + '-' + (os.path.splitext(filename))[0] + '.ann'
    outpath = 'model-annotations/' + choice + '/' + outname
    write_entities(text, entity_list, outpath)
    #print('finished')

def annotate_with_dbpedia(filepath, choice='default'):
    # choices are 'default', 'uncased', or 'truecased'
    # if choice=='truecased', then the spaCy uncased model is called to truecase the named entities before the text is sent to dbpedia

    def find_dbpedia_type(ent):
        # select only interested types of entities
        prefix = 'DBpedia:'
        interested_types = ['Person','Place','Organisation','Device']
        try:
            types_list = ent._.dbpedia_raw_result['@types'].split(',')
            for category in interested_types:
                if (prefix+category) in ent._.dbpedia_raw_result['@types']:
                    return category
        except:
            return None
        return None
    
    infile = open(filepath, 'r')
    text = infile.read()
    infile.close()

    if(choice=='uncased'):
        text = text.lower()
    elif(choice=='truecased'):
        text = text.lower()
        text = truecase(text)

    # Load small English core model
    nlp = spacy.blank('en')
    # add the dbpedia_spotlight pipeline stage
    nlp.add_pipe('dbpedia_spotlight')

    # since dbpedia_spotlight calls outside database, there is a slim chance that it will fail \
    # we will loop here until it succeeds
    calling_success = False
    while(calling_success == False):
        try:
            doc = nlp(text)
            calling_success = True
        except:
            pass
        
    #print('detecting named entities..')
    #print('annotating..')

    tok_idx = {}
    for (n, tok) in enumerate(doc):
        p1 = tok.idx
        p2 = p1 + len(tok.text)
        tok_idx[n] = (p1, p2)

    entity_list = [[find_dbpedia_type(ent), tok_idx[ent.start][0], tok_idx[ent.end - 1][1]] for ent in doc.ents if find_dbpedia_type(ent)!=None]
    entity_list.sort(key=lambda x: x[2]) # sorted by end character's index
    bad_list = find_bad_indices(filepath) # find 'bad indices' (i.e. before the first ':' of each line in the transcript)
    entity_list = remove_bad_indices(bad_list, entity_list) # remove entities in the 'bad indices' ranges

    # write entities to output file 
    filename = os.path.basename(filepath)
    outname = (os.path.splitext(filename))[0] + '.ann'
    if(choice == 'default'):
        outpath = 'dbpedia-annotations/cased/' + outname
    else:
        outpath = 'dbpedia-annotations/'+choice+'/' + outname
    write_entities(text, entity_list, outpath)
    #print('finished')

def annotate_all_transcripts():
    text_dir = "transcripts"
    for choice in ['default','uncased','force-uncased']:
        for text_name in os.listdir(text_dir):
            if(text_name.endswith(".txt")):
                text_path = (text_dir + "/" + text_name)
                print("annotating "+text_name+" (choice = "+choice+")")
                annotate(text_path, choice)

def annotate_all_transcripts_with_dbpedia():
    text_dir = "transcripts"
    #for choice in ['default','uncased','truecased']:
    for choice in ['truecased']:
        for text_name in os.listdir(text_dir):
            if(text_name.endswith(".txt")):
                text_path = (text_dir + "/" + text_name)
                print("annotating "+text_name+" (choice = "+choice+")")
                annotate_with_dbpedia(text_path, choice)

def clean_all_gold_annotations():
    # this function read gold annotation files and clean them of 'bad indices'
    before_clean_dir = "annotations-before-cleaned"
    after_clean_dir = "annotations"
    text_dir = "transcripts"
    for ann_name in os.listdir(before_clean_dir):
        if(ann_name.endswith(".ann")):
            text_path = text_dir + '/' + (os.path.splitext(ann_name))[0] + '.txt'
            if(os.path.isfile(text_path)):
                print("cleaning "+ann_name)
                bad_list = find_bad_indices(text_path)
                ann_path = (before_clean_dir + "/" + ann_name)
                infile = open(ann_path, 'r')
                lines = infile.readlines()
                infile.close()
                entity_list = [line.split() for line in lines]
                entity_list = [[e[1], int(e[2]), int(e[3]), " ".join(e[4:])] for e in entity_list] # change (start, end) from str to int
                entity_list.sort(key=lambda x: x[2]) # sorted by end character's index
                entity_list = remove_bad_indices(bad_list, entity_list) # remove entities in the 'bad indices' ranges

                # write entities to output file
                outpath = after_clean_dir + '/' + ann_name
                outfile = open(outpath, 'w')
                for i, [label, start, end, ent_text] in enumerate(entity_list):
                    outfile.write(('T'+str(i+1)) + '\t' + label + ' ' + str(start) + ' ' + str(end) + '\t' + ent_text + '\n')
                outfile.close()

def find_intersection(filepath1, filepath2, outpath, overlap_choice=False):
    # this function read two annotation files and output another annotation file which contains only entities that exist in both files.
    # By 'exist in both files', the entity's (start, end) indices must be the same in both files, but the types may be different.
    # However, if overlap_choice == True, the (start, end) indices need not be the same, they just need to overlap.

    def lines_to_list(lines, metric='token'):
        entity_list = [tuple(line.split()) for line in lines] 
        entity_list = [(l[1],int(l[2]),int(l[3]), " ".join(l[4:])) for l in entity_list]
        return entity_list

    def overlap(entity_set, e2_range): # tuple parameter is not supported in Python 3. So I have to unpack it
        (start_e2, end_e2) = e2_range
        # check if e2, which spans the (start_e2, end_e2) range, overlaps with some entity in the entity_set
        for (start_e1, end_e1) in entity_set:
            if((start_e1 < end_e2) and (end_e1 > start_e2)): # in other words, (start_e1 <= (end_e2-1)) and ((end_e1-1) >= start_e2)
                return True
        return False
    
    infile = open(filepath1, 'r')
    entity_list = lines_to_list(infile.readlines())
    infile.close()
    entity_set = set([(e[1],e[2]) for e in entity_list]) # the set of (start, end) of all entities in filepath1
    infile = open(filepath2, 'r')
    entity_list = lines_to_list(infile.readlines())
    infile.close()

    if(overlap_choice == False):
        entity_list = [e for e in entity_list if (e[1],e[2]) in entity_set]
    else:
        entity_list = [e for e in entity_list if overlap(entity_set, (e[1],e[2]))]

    outfile = open(outpath, 'w')
    for i, [label, start, end, ent_text] in enumerate(entity_list):
        outfile.write(('T'+str(i+1)) + '\t' + label + ' ' + str(start) + ' ' + str(end) + '\t' + ent_text + '\n')
    outfile.close()

def find_intersection_all():
    # this function reads spacy-annotated files and intersects them with dbpedia-annotated files
    for overlap_choice in [False, True]:
        for dbpedia_choice in ['truecased']: # cased, truecased
            dbpedia_dir = 'dbpedia-annotations/' + dbpedia_choice
            if(overlap_choice):
                out_dir = 'dbpedia-annotations/' + dbpedia_choice + '-intersected-relaxed'
            else:
                out_dir = 'dbpedia-annotations/' + dbpedia_choice + '-intersected-strict'
            if(dbpedia_choice == 'cased'):
                spacy_dir = 'model-annotations/default'
                spacy_prefix = 'default-'
            else:
                spacy_dir = 'model-annotations/uncased'
                spacy_prefix = 'uncased-'
            for dbpedia_ann_name in os.listdir(dbpedia_dir):
                if(dbpedia_ann_name.endswith(".ann")):
                    spacy_ann_name = (spacy_prefix + dbpedia_ann_name)
                    spacy_path = spacy_dir + '/' + spacy_ann_name
                    if(os.path.isfile(spacy_path)):
                        dbpedia_path = dbpedia_dir + '/' + dbpedia_ann_name
                        outpath = out_dir + '/' + dbpedia_ann_name
                        print("creating intersected annotation at "+outpath)
                        find_intersection(spacy_path, dbpedia_path, outpath, overlap_choice)
            

if __name__ == '__main__':

    #annotate('transcripts/cpb-aacip-507-154dn40c26-transcript.txt')
    #annotate_all_transcripts()
    #clean_all_gold_annotations()
    #annotate_all_transcripts_with_dbpedia()
    #find_intersection_all()
    clean_all_transcripts()
    
