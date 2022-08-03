import sys
import os
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
        while(entity_list[i1][1] < bad_list[i2][2]): # [1] = start index of the entity, [2] = end index
            if(entity_list[i1][2] > bad_list[i2][1]): # the entity pointed by i1 overlaps with the forbidden indices
                entity_list.pop(i1)
            else:
                i1 += 1
        i2 += 1
    return entity_list

def truecase(text):
    # this function is currently unused
    ner = spacy.load("ner_models/model-best-uncased-sm")
    doc = ner(text.lower())

    tok_idx = {}
    for (n, tok) in enumerate(doc):
        p1 = tok.idx
        p2 = p1 + len(tok.text)
        tok_idx[n] = (p1, p2)

    interested_labels = ('ORG','PERSON','GPE','LOC')
    text_list = list(text.lower()) # python string is immutable, but we want to modify input_text
    for (n, ent) in enumerate(doc.ents):
        if(ent.label_ in interested_labels):
            start = tok_idx[ent.start][0]; end = tok_idx[ent.end - 1][1]
            entity_text = text[start:end]
            for m in re.finditer(r'\S+', entity_text):
                capitalized_index = (start + m.start())
                text_list[capitalized_index] = text_list[capitalized_index].upper()
    text = "".join(text_list)
    print(text[:1000])
    return text

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

    interested_labels = ('ORG','NORP','PERSON','GPE','LOC','FAC','PRODUCT','EVENT','WORK_OF_ART') # must include 'PER' if trained conll-style
    entity_list = [[ent.label_, tok_idx[ent.start][0], tok_idx[ent.end - 1][1]] for ent in doc.ents if ent.label_ in interested_labels]
    # each sublist in entity_list is [label, start, end]
    entity_list.sort(key=lambda x: x[2]) # sorted by end character's index
    bad_list = find_bad_indices(filepath) # find 'bad indices' (i.e. before the first ':' of each line in the transcript)
    entity_list = remove_bad_indices(bad_list, entity_list) # remove entities in the 'bad indices' ranges

    # write entities to output file 
    filename = os.path.basename(filepath)
    outname = choice + '-' + (os.path.splitext(filename))[0] + '.ann'
    outpath = 'model-annotations/' + choice + '/' + outname
    outfile = open(outpath, 'w')
    for i, [label, start, end] in enumerate(entity_list):
        outfile.write(('T'+str(i+1)) + '\t' + label + ' ' + str(start) + ' ' + str(end) + '\t' + repr(text[start:end])[1:-1] + '\n')
        # repr(ent.text)[1:-1] prints newline character as the substring '\n'
    outfile.close()
    #print('finished')

def annotate_all_transcripts():
    text_dir = "transcripts"
    for choice in ['default','uncased','force-uncased']:
        for text_name in os.listdir(text_dir):
            if(text_name.endswith(".txt")):
                text_path = (text_dir + "/" + text_name)
                print("annotating "+text_name+" (choice = "+choice+")")
                annotate(text_path, choice)

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

if __name__ == '__main__':

    #annotate('transcripts/cpb-aacip-507-154dn40c26-transcript.txt')
    #annotate_all_transcripts()
    clean_all_gold_annotations()
    
