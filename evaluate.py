import os
import sys
from collections import defaultdict

gold_dir = "annotations"
test_dir_base = "model-annotations"

label_dict = {'PERSON':'person', 'ORG':'organization', 'FAC':'location', 'GPE':'location', 'LOC':'location'}
interested_labels = list(set(label_dict.values()))

def sum_over_labels(confusion_dict):
    for s in ['true_pos','false_neg','false_pos']:
        confusion_dict[(s,'all')] = sum([confusion_dict[(s,label)] for label in interested_labels])

def calculate_F1(confusion_dict):
     all_labels = (interested_labels + ['all'])
    for label in all_labels:
        try:
            confusion_dict[('precision',label)] = (confusion_dict[('true_pos',label)] / (confusion_dict[('true_pos',label)] + confusion_dict[('false_pos',label)]))
        except ZeroDivisionError:
            confusion_dict[('precision',label)] = None
        try:
            confusion_dict[('recall',label)] = (confusion_dict[('true_pos',label)] / (confusion_dict[('true_pos',label)] + confusion_dict[('false_neg',label)]))
        except ZeroDivisionError:
            confusion_dict[('recall',label)] = None
        if((confusion_dict[('precision',label)] != None) and (confusion_dict[('recall',label)] != None)):
            confusion_dict[('F1',label)] = ((2 * confusion_dict[('precision',label)] * confusion_dict[('recall',label)]) / \
                                        (confusion_dict[('precision',label)] + confusion_dict[('recall',label)]))
        else:
            confusion_dict[('F1',label)] = None

    gold_entity_count = (confusion_dict[('true_pos',label)] + confusion_dict[('false_neg',label)])
    test_entity_count = (confusion_dict[('true_pos',label)] + confusion_dict[('false_pos',label)])
    if(confusion_dict[('F1','all')]!=None):
        print('precision\t'+('%.3f' % confusion_dict[('precision','all')])+' ('+str(confusion_dict[('true_pos',label)])+'/'+str(test_entity_count)+')')
        print('recall\t'+('%.3f' % confusion_dict[('recall','all')])+' ('+str(confusion_dict[('true_pos',label)])+'/'+str(gold_entity_count)+')')
        print('F1\t'+('%.3f' % confusion_dict[('F1','all')]))
    else:
        print("F1 can't be calculated because the number of false positive, or false negative, or both, is zero.")

def evaluate_strict(goldlist, testlist, label_choice):
    """ Here, the named entities' spans, or tokens, in the gold and test file must strictly match to count as True Positive.
    The algorithm implemented here is that: First, the entities from the gold file will be put into dictionary (with their
    (start_index, end_index) tuple as the dict's key). Then the entities from the test file will be checked whether they
    find their matchs in the dictionary. """

    def same_label(e1_0, e2_0):
        if((e1_0 == e2_0) or ((e1_0 == 'organization') and (e2_0 == 'location') and (label_choice == 'LOC_to_ORG')) \
           or (label_choice == 'blind')):
            return True
        return False

    #print("number of real entities:",len(goldlist))
    #print("number of proposed entities:",len(testlist))
    confusion_dict = defaultdict(lambda: 0)
    gold_dict = { (start, end): label for (label, start, end, ent_text) in goldlist}
    for (test_label, start, end, ent_text) in testlist:
        if((start, end) in gold_dict):
            gold_label = gold_dict[(start, end)]
            if(same_label(gold_label, test_label)):
                confusion_dict[('true_pos', gold_label)] += 1
                gold_dict.pop((start, end))
            else:   # the model get the span right but the label wrong
                confusion_dict[('false_pos', test_label)] += 1
        else:
            confusion_dict[('false_pos', test_label)] += 1

    for (start, end) in gold_dict: # iterate over tbe spans in gold file that are not matched by those in test file
        gold_label = gold_dict[(start, end)]
        confusion_dict[('false_neg', gold_label)] += 1
    sum_over_labels(confusion_dict)
        
    return confusion_dict
            

def evaluate_relaxed(goldlist, testlist, label_choice):
    """ This method represents a metric that count named entities' spans in the gold and test file as True Positive if 
    at least one of the two is a substring of the other. For example, the gold file may contain 'CEO Mark Zuckerburg'
    and the test file 'Mark Zuckerburg' or vise versa. The algorithm implemented here is inspired by mergesort, whereas
    the first elements of goldlist and testlist (i.e. the two entities to be compared) are compared with each other, then
    one or both of the lists will be popped (from the front) until the lists are empty. """

    #print("number of real entities:",len(goldlist))
    #print("number of proposed entities:",len(testlist))
    confusion_dict = defaultdict(lambda: 0)
    while(len(goldlist)>0 and len(testlist)>0):
        e1 = goldlist[0]; e2 = testlist[0]
        # for each e, e[0] is the label, e[1] is start index, e[2] is end index
        def contain(e1, e2): # check if one entity can be contained inside another
            if(((e1[1]>=e2[1]) and (e1[2]<=e2[2])) or ((e2[1]>=e1[1]) and (e2[2]<=e1[2]))):
                return True
            return False

        def same_label(e1_0, e2_0):
            if((e1_0 == e2_0) or ((e1_0 == 'organization') and (e2_0 == 'location') and (label_choice == 'LOC_to_ORG')) \
               or (label_choice == 'blind')):
                return True
            return False
            
        if(same_label(e1[0], e2[0]) and contain(e1, e2)):
            confusion_dict[('true_pos',e1[0])] += 1
            goldlist.pop(0); testlist.pop(0)
        elif(e1[2]<e2[2]):
            confusion_dict[('false_neg',e1[0])] += 1
            goldlist.pop(0)
        else:
            confusion_dict[('false_pos',e2[0])] += 1
            #print(1,e2[0],text[e2[1]:e2[2]],e2[1],e2[2])
            testlist.pop(0)
    while(len(goldlist)>0):
        confusion_dict[('false_neg',goldlist[0][0])] += 1
        #print(2,goldlist[0][0],text[goldlist[0][1]:goldlist[0][2]],goldlist[0][1],goldlist[0][2])
        goldlist.pop(0)
    while(len(testlist)>0):
        confusion_dict[('false_pos',testlist[0][0])] += 1
        testlist.pop(0)
    sum_over_labels(confusion_dict)
        
    return confusion_dict

def evaluate(gold_path, test_path, label_choice, metric):

    # the lines_to_list method converts lines in .ann file to a list of annotations sorted by end character's index.
    # the lines in the annotation files are supposed to be already sorted this way.
    # This method also change the test file's labels to match the gold file's terminology if change_label is True.
    # if metric == 'token', each entity is tokenized before being put into entity_list.
    def lines_to_list(lines, change_label=False, metric='token'):
        entity_list = [tuple(line.split()) for line in lines] 
        if(change_label):
            entity_list = [(label_dict[l[1]],int(l[2]),int(l[3]), " ".join(l[4:])) for l in entity_list if l[1] in label_dict]
        else:
            entity_list = [(l[1],int(l[2]),int(l[3]), " ".join(l[4:])) for l in entity_list if l[1] in interested_labels]
        if(metric == 'token'): # each entity must be tokenized
            i = 0
            while(i < len(entity_list)): # can't use a for loop since len(entity_list) can increase
                (label, start, end, ent_text) = entity_list[i]
                if(' ' in ent_text):
                    ent_text_new = ent_text.split(' ')[0]
                    next_ent_text = ' '.join(ent_text.split(' ')[1:])
                    entity_list.pop(i)
                    entity_list.insert(i, (label, start, start+len(ent_text_new), ent_text_new))
                    entity_list.insert(i+1, (label, start+len(ent_text_new)+1, end, next_ent_text))
                i += 1
        return entity_list
    
    infile = open(gold_path, 'r')
    goldlist = lines_to_list(infile.readlines())
    infile.close()
    #print(goldlist)

    infile = open(test_path, 'r')
    testlist = lines_to_list(infile.readlines(), change_label=True)
    infile.close()
    #print(testlist)
    
    confusion_dict = defaultdict(lambda: 0)
    if(metric == 'token' or metric == 'strict'): # e1 and e2 must strictly match, whether at a token level or at a span level
        confusion_dict = evaluate_strict(goldlist, testlist, label_choice)
    elif(metric == 'relaxed'): # e1 and e2 do not have to completely overlapped, so the evaluation algorithm is more complex
        confusion_dict = evaluate_relaxed(goldlist, testlist, label_choice)
    else:
        print("ERROR: parameter 'metric' has invalid value")
        sys.exit()
    return confusion_dict

def evaluate_all(annotation_choice, label_choice, metric = 'token'):
    confusion_dict_all = defaultdict(lambda: 0)
    for label in interested_labels + ['all']:
        for s in ['true_pos','false_pos','false_neg']:
            confusion_dict_all[(s, label)] = 0
    test_dir = (test_dir_base + "/" + annotation_choice)
    for gold_name in os.listdir(gold_dir):
        if(gold_name.endswith(".ann")):
            test_name = (annotation_choice + "-" + gold_name)
            test_path = (test_dir + "/" + test_name)
            if(os.path.isfile(test_path)):
                gold_path = (gold_dir + "/" + gold_name)
                #print("evaluating "+test_name)
                confusion_dict = evaluate(gold_path, test_path, label_choice, metric)
                #print(confusion_dict)
                confusion_dict_all = {k: confusion_dict_all[k] + confusion_dict[k] for k in set(confusion_dict_all)}
    #print(confusion_dict_all)
    calculate_F1(confusion_dict_all)

def evaluate_test(annotation_choice, label_choice, metric = 'token'):
    # use only for debugging the program
    test_dir = (test_dir_base + "/" + annotation_choice)
    gold_name = 'cpb-aacip-507-154dn40c26-transcript.ann'
    test_name = (annotation_choice + "-" + gold_name)
    test_path = (test_dir + "/" + test_name)
    if(os.path.isfile(test_path)):
        gold_path = (gold_dir + "/" + gold_name)
        print("evaluating "+test_name)
        confusion_dict = evaluate(gold_path, test_path, label_choice, metric)
        calculate_F1(confusion_dict)

if __name__ == '__main__':

    # annotation_choice are from ['default','uncased','force-uncased']
    # label_choice are from ['strict', 'LOC_to_ORG', 'blind']
    # metric are from ['token','strict','relaxed']

    #evaluate_test(annotation_choice='force-uncased', label_choice='LOC_to_ORG', metric = 'token')
    evaluate_all(annotation_choice='uncased', label_choice='LOC_to_ORG')
    
