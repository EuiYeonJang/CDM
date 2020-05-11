import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle
import copy
from tqdm import tqdm
from collections import Counter
""" Here, we process the ICSI files into data structures usable for our analysis.
The following file types are in the ICSI corpus:

Dialogue acts files, e.g. Bdb001.A.dialogue-acts.xml
    - Contain a list of dialogue acts for each (part of) a meeting
    - Meeting number is indicated by Bdb001
    - Part of meeting is defined by A

    - File contains the nite:root structure
        - This contains many <dialogueact> structures
            - Each dialogue act has a nite:id - id of dialogue act
            - start time
            - end time
            - type (this is a tag corresponding to the function of the dialogue act)
            - adjacency (this is a tag corresponding to the adjacency pair this dialogue act belongs to - save as string now, and process later)
            - original type (what's this? not clear, but we will save it anyway
            - participant
            - A child structure, which refers to the words, as in "Bdb001.A.words.xml#id(Bdb001.w.701)..id(Bdb001.w.702)"
                - We should decompose this to
                - File does not need to be listed, since word indices are unique - file: Bdb001.A.words.xml (or file_id: Bdb001.A.words)
                - and word_index_start
                - word_index_end
Segment files, e.g. Bdb001.A.segs.xml
    - These files contain info on 'segments', which are periods of spoken speech. Similar to dialogue acts, except not 
    with a similarly specific annotation. 
    - File contains nite: root structure
        - contains <segment> structure, which lists only the speaker and times.
    - Do not use these files, all necessary info should be in the dialogue acts

Words files, e.g. Bdb001.A.words.xml
    - contain <nite:root> structure
        - This contains many <vocalsound>, <nonvolcalsound>, <w>, <disfmarker>, <comment> and <pause> objects
        - We only need the word objects
        - these contain as tags:
            - start time
            - end time
            - c
            - k
            - qut
            - t
        - and as content:
            - A single word, or punctuation mark
    - convert this to a dict of words, with each key being the id

Speakers file, speakers.xml

"""


class Word():
    def __init__(self):
        self.id = ''
        self.text = ''
        self.start_time = ''
        self.end_time = ''
        self.c = ''
        self.t = None
        self.k = None
        self.qut = None

    def __init__(self, id, text, start_time, end_time, c, t = None, k = None, qut = None):
        self.id = id
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.c = c
        self.t = t
        self.k = k
        self.qut = qut

class Speaker():
    def __init__(self, tag, gender, age, education):
        self.tag = tag # This is the speaker id
        self.gender = gender
        self.age = age
        self.education = education

class DialogueAct():
    # TODO: Include calling of text and participant functions in initialisation
    def __init__(self, id, start_time, end_time, participant_id, type_, original_type, channel, 
    comment, adjacency, start_word_id, end_word_id, words, speakers):
        self.id = id
        self.start_time = start_time
        self.end_time = end_time
        self.participant_id = participant_id
        # self.participant = None
        self.type_ = type_
        self.original_type = original_type
        self.channel = channel
        self.comment = comment
        self.adjacency = adjacency
        self.start_word_id = start_word_id
        self.end_word_id = end_word_id
        # self.text = None

        self.text = self.retrieve_text(words)
        self.participant = self.retrieve_participant(speakers)
        # self.adjacency_dict = self.list_adjacency()
    
    def retrieve_text(self, words_dict):
        # retrieve index in words_dict of start_word_id
        start_word_index = list(words_dict.keys()).index(self.start_word_id)
        if self.end_word_id is not None:
            end_word_index = list(words_dict.keys()).index(self.end_word_id)+1
        else:
            end_word_index = start_word_index+1

        act_words = list(words_dict.values())[start_word_index:end_word_index]
        string = ' '.join([word.text for word in act_words if word.text])
        # self.text = string
        print(string)
        return string
    
    def retrieve_participant(self, speakers):
        participant_id = self.participant_id
        participant = speakers[participant_id]
        print(participant.tag)
        return participant

    def list_adjacency(self):
    # also call a function to split the adjacency tags into (1) a list with all tags, and (2) a list of the number of plusses
    # Or we make it a dict! tag: (a/b, number_of_plusses, dash_number)
    # A dash indicates a split (so multiple equal utterances in the adjacency pair, by different speakers)
    # A plus indicates another utterance with the same function, by one of the same speakers. (So if the replier utters two 
    # utterances in a row, the second gets a plus)
    # Then use these tags to create a set of all adjacency tags.
        # print(self.adjacency)
        if self.adjacency is not None:
            adjacencies = self.adjacency.split('.')
            print(adjacencies)
            # There is also a dash - 
            adjacency_dict = {}
            for tag in adjacencies:
                print("Tag: ", tag)
                if (tag.count('a') + tag.count('b')) != 1:
                    continue
                # Select the first a or b letter, this ends the adjacency pair tag
                letter_index = max(tag.find('a'), tag.find('b')) # Removes -1 from unfound letter
                final_tag = tag[:letter_index]
                letter = tag[letter_index]
                number = 1
                print("final tag: ", final_tag, letter_index)
                if '-' in tag:
                    number = tag[letter_index+2]
                
                no_of_plusses = tag.count('+')

                adjacency_dict[final_tag] = (letter, number, no_of_plusses)

            return adjacency_dict

        else:
            return None


# class DialogueTurn():
    # this class should contain the 
    #  dialogue act id 
    #  all Word objects in this dialogue act
    #  a string of the complete turn (so combination of all Word.text)
    #  participant id
    #  and in some way a reference to all adjacency pairs. We could also extract this recursively.  l

# class Adjacencyquence():
    # An object of this class should contain, for each adjacency pair: 
    # The adjacency pair id
    # A list of utterances that are part of the same sequence of adjacency pairs. 
    # A list with ranking (no plusses is zero, 1 plus is 1, etc..)
    # original: original utterance, which elicits a response (DialogueAct object)
    # original_text: words of original utterance
    # response: replying utterance (DialogueAct object)
    # utterances: |

def main():
    print("Extracting ICSI..")

    this_file_path = os.path.dirname(os.path.abspath(__file__))
    relative_ICSI_path = '/../Corpora/ICSI/'
    ICSI_path = this_file_path + relative_ICSI_path
    parsed_words_path = ICSI_path + 'parsed_words.pkl'
    parsed_acts_path = ICSI_path + 'parsed_acts.pkl'
    parsed_adjacency_path = ICSI_path + 'adjacency_dict_1.pkl'
    adjacency_pair_list_pickle = ICSI_path + 'adjacency_list.pkl'
    parsed_speakers_path = ICSI_path + 'parsed_speakers.pkl'
    words_directory = ICSI_path + 'Words/'
    acts_directory = ICSI_path + 'DialogueActs/'
    speakerspath = ICSI_path + 'speakers.xml'


    if not os.path.isfile(parsed_words_path):
        print("Extracting Words files..")
        words = {}
        for subdir, dirs, files in os.walk(words_directory):
            for file in files:
                filepath = subdir + os.sep + file

                if filepath.endswith('.xml'):
                    tree = ET.parse(filepath)
                    root = tree.getroot()

                    for child in root:
                        if child.tag == 'w':
                            # print(child.attrib)
                            id = child.attrib['{http://nite.sourceforge.net/}id']
                            start_time = child.attrib.get('starttime', None)
                            end_time = child.attrib.get('endtime', None)
                            c = child.attrib.get('c', None)
                            t = child.attrib.get('t', None)
                            k = child.attrib.get('k', None)
                            qut = child.attrib.get('qut', None)
                            text = child.text

                            words[id] = Word(id, text, start_time, end_time, c, t = t, k = k, qut = qut)
                        else:
                            id = child.attrib['{http://nite.sourceforge.net/}id']
                            start_time = child.attrib.get('starttime', None)
                            end_time = child.attrib.get('endtime', None)
                            c = child.attrib.get('c', None)
                            t = child.attrib.get('t', None)
                            k = child.attrib.get('k', None)
                            qut = child.attrib.get('qut', None)
                            text = False
                            words[id] = Word(id, text, start_time, end_time, c, t = t, k = k, qut = qut)


        with open(parsed_words_path, 'wb') as parsed_words_file:
            pickle.dump(words, parsed_words_file)
        # Make sure to import Word class when unpickling.
    else:
        with open(parsed_words_path, 'rb') as parsed_words_file:
            words = pickle.load(parsed_words_file)

    print(list(words.values())[0])
    print(len(words))

    # Parse speakers
    if not os.path.isfile(parsed_speakers_path):
        print("Extracting speakers file..")

        tree = ET.parse(speakerspath)
        root = tree.getroot()
        speakers = {}

        for child in root:
            # print(child.tag, child.attrib)
            # print(child.attrib['tag'])
            tag = child.attrib['tag']
            gender = child.attrib.get('gender', None)
            # print(child['age'].text)
            for grandchild in child:
                # print(grandchild.tag, granchild.attrib)
                if grandchild.tag == 'age':
                    age = grandchild.text
                elif grandchild.tag == 'education':
                    education = grandchild.text
            speakers[tag] = Speaker(tag, gender, age, education)
            # print(tag, gender)


        with open(parsed_speakers_path, 'wb') as parsed_speakers_file:
            pickle.dump(speakers, parsed_speakers_file)
    else:
        with open(parsed_speakers_path, 'rb') as parsed_speakers_file:
            speakers = pickle.load(parsed_speakers_file)

    print(list(speakers.values())[0])
    print(len(speakers))

    if not os.path.isfile(parsed_adjacency_path):
        print("Extracting dialogue acts..")
        # TODO: Implement looping over all dialogue act files
        # filepath = ICSI_path + 'DialogueActs/Bdb001.A.dialogue-acts.xml'
        dialogue_acts = {}
        adjacency_dict = {}
        
        for subdir, dirs, files in tqdm(os.walk(acts_directory)):
            for file in tqdm(files):
                filepath = subdir + os.sep + file

                if filepath.endswith('.xml'):

                    tree = ET.parse(filepath)
                    root = tree.getroot()
                    for child in tqdm(root):
                        # print(child.tag, child.attrib, child.text)

                        id = child.attrib['{http://nite.sourceforge.net/}id']
                        start_time = child.attrib.get('starttime', None)
                        end_time = child.attrib.get('endtime', None)
                        type_ = child.attrib.get('type', None)
                        adjacency = child.attrib.get('adjacency', None)
                        original_type = child.attrib.get('original_type', None)
                        channel = child.attrib.get('channel', None)
                        comment = child.attrib.get('comment', None)
                        participant_id = child.attrib.get('participant', None)
                        # print("Participant id: ", participant_id)

                        for grandchild in child:
                            href = grandchild.attrib.get('href', None)
                            opening_bracket_index = href.find('(')
                            word_ids = href[opening_bracket_index+1:]
                            first_closing_bracket_index = word_ids.find(')')

                            start_word_id = word_ids[:first_closing_bracket_index]
                            end_word_id = word_ids[first_closing_bracket_index:]
                            end_opening_bracket = end_word_id.find('(')
                            if end_opening_bracket != -1:
                                end_word_id = end_word_id[end_opening_bracket+1:-1]
                            else:
                                end_word_id = None
                                


                        # dialogue_acts[id] = DialogueAct(id, start_time, end_time, participant_id, type_, \
                            # original_type, channel, comment, adjacency, start_word_id, end_word_id, words=words, speakers=speakers)


                        if adjacency is not None:
                            adjacencies = adjacency.split('.')
                            # print(adjacencies)
                            # There is also a dash - 
                            
                            # text = words[start_word_id:end_word_id] # To retrieve words, convert the keys to indices in with list(words.keys), and then select the Words between those indices
                            # text_start_index = list(words.keys()).index(start_word_id)
                            # text_end_index = list(words.keys()).index(end_word_id)

                            # text = 

                            # retrieve index in words_dict of start_word_id
                            # TODO: COnvert this to function
                            start_word_index = list(words.keys()).index(start_word_id)
                            if end_word_id is not None:
                                end_word_index = list(words.keys()).index(end_word_id)+1
                            else:
                                end_word_index = start_word_index+1

                            act_words = list(words.values())[start_word_index:end_word_index]
                            string = ' '.join([word.text for word in act_words if word.text])
                            # self.text = string
                            # print(string)
                            # return string

                            for tag in adjacencies:
                                # print("Tag: ", tag)
                                # if tag == '5a':
                                #     print("Tag is 5a!")
                                if (tag.count('a') + tag.count('b')) != 1:
                                    continue # This filters out faulty tags (sadly these exist in the dataset)
                                # Select the first a or b letter, this ends the adjacency pair tag
                                letter_index = max(tag.find('a'), tag.find('b')) # Removes -1 from unfound letter
                                final_tag = file[:6] + tag[:letter_index] # Tags are reused in different meetings, this splits them up
                                letter = tag[letter_index]
                                number = 1
                                # print("final tag: ", final_tag, letter_index)
                                if '-' in tag:
                                    number = tag[letter_index+2]
                                
                                no_of_plusses = tag.count('+')
                                # insert_dict = {letter: {number: [string_list, participant_id]}} 
                                # Check if there is already an entry. If not, add string_list with length of current number of plusses
                                # If there is one, add it on the index of the list that corresponds to the number of plusses
                                
                                # tupl = (letter, number, no_of_plusses, string, participant_id)
                                # insert_dict = {letter: (number, no_of_plusses, string_list, participant_id)}
                                # adjacency_dict[letter][number]
                                if final_tag in adjacency_dict.keys() and letter in adjacency_dict[final_tag].keys():
                                    # if letter in adjacency_dict[final_tag].keys():
                                    for item_index, item in enumerate(adjacency_dict[final_tag][letter]): # loop through all existing acts for the current tag
                                        if number == item[0] and participant_id == item[3]:
                                            # if no_of_plusses == item[2] + 1:
                                                # text = item[3] + ' ' + string
                                            text = item[2]
                                            if len(text) < no_of_plusses + 1:
                                                text += ['']*(no_of_plusses + 1 - len(text))
                                            text[no_of_plusses] = string
                                            insert_tupl = (number, no_of_plusses, text, participant_id)
                                            adjacency_dict[final_tag][letter][item_index] = insert_tupl
                                            # print(text)
                                            break
                                        # elif no_of_plusses < item[2]:
                                            # print("\n Less plusses than seen before \n")
                                        # else:
                                            # print("Skipped a plus..", tag)

                                # adjacency_dict[final_tag]
                                    else:
                                        string_list = ['']*(no_of_plusses+1)
                                        string_list[no_of_plusses] = string
                                        # tupl = (letter, number, no_of_plusses, string_list, participant_id)
                                        insert_tupl = (number, no_of_plusses, string_list, participant_id)
                                        adjacency_dict[final_tag][letter].append(insert_tupl)
                                        # print(string_list)
                                elif final_tag in adjacency_dict.keys(): # Only the letter is not in there yet. We need to insert it

                                    string_list = ['']*(no_of_plusses+1)
                                    string_list[no_of_plusses] = string
                                    # tupl = (letter, number, no_of_plusses, string_list, participant_id)
                                    insert_tupl = (number, no_of_plusses, string_list, participant_id)
                                    adjacency_dict[final_tag][letter] = [insert_tupl]
                                else: # So final_tag is not in there yet. We need to insert it
                                    string_list = ['']*(no_of_plusses+1)
                                    string_list[no_of_plusses] = string
                                    # tupl = (letter, number, no_of_plusses, string_list, participant_id)
                                    insert_dict = {letter: [(number, no_of_plusses, string_list, participant_id)]}
                                    adjacency_dict[final_tag] = insert_dict
                                    # print(string_list)
                        # dialogue_acts[id].retrieve_text(words)
                        # dialogue_acts[id].retrieve_participant(speakers)
                # break


        with open(parsed_adjacency_path, 'wb') as parsed_acts_file:
            pickle.dump(adjacency_dict, parsed_acts_file)
    else:
        with open(parsed_adjacency_path, 'rb') as parsed_acts_file:
            adjacency_dict = pickle.load(parsed_acts_file)

    # print(list(adjacency_dict.values()))
    print(len(adjacency_dict))
    
    # del speakers
    # del words
    # print(adjacency_dict['5'][''])
    if not os.path.isfile(adjacency_pair_list_pickle):
        adjacency_pairs = []
        # Now we sould convert the 'text' entries to counters, and separate each a-b pair.
        for AP_tag in adjacency_dict.keys():
            print(AP_tag, adjacency_dict[AP_tag])
            # this contains a list of dialogue acts that belong to this one adjacency pair
            for dialogue_act in adjacency_dict[AP_tag].get('a', []):
                a_counter = Counter()
                text = dialogue_act[2]
                for utterance in text:
                    list_of_tokens = utterance.lower().split(' ')
                    a_counter.update(list_of_tokens)

                a_participant_gender = dialogue_act[3][0]
                for dialogue_act in adjacency_dict[AP_tag].get('b', []):
                    print("Found b!")
                    b_counter = Counter()
                    text = dialogue_act[2]
                    for utterance in text:
                        list_of_tokens = utterance.lower().split(' ')
                        b_counter.update(list_of_tokens)
                    b_participant_gender = dialogue_act[3][0]
                    dic = {
                        'a': 
                            {   'counter': a_counter, 
                                'gender': a_participant_gender
                            },
                        'b': 
                            {   'counter': b_counter, 
                                'gender': b_participant_gender
                            }   
                            }
                    adjacency_pairs.append(dic)

        with open(adjacency_pair_list_pickle, 'wb') as adjacency_list_file:
            pickle.dump(adjacency_pairs, adjacency_list_file)
    else:
        with open(adjacency_pair_list_pickle, 'rb') as adjacency_list_file:
            adjacency_pairs = pickle.load(adjacency_list_file)

    print(adjacency_pairs[0:2])
    # The adjacency_list has a dict of each single adjacency pair at each list entry
    # This dict contains the first turn at key 'a', and the second at turn 'b'.


        # dialogue_acts = adjacency_dict[AP_tag]
        # for dialogue_act in dialogue_acts:
            # We should now extract a-b pairs, such that all possible pairs are formed. 
            # Would be nice to loop over the a's, and then over the b's.
        # print(adjacency_dict[AP_tag])

    # example_act = list(dialogue_acts.values())[0]
    # text = example_act.text(words)
    # print("Example text: ", text)

    # Now extract all keys that are in the adjacency pair. Put in each tuple into a list, (then order?)
    # pairs = {}
    # for act in tqdm(dialogue_acts):
    #     for key in dialogue_acts.keys():
    #         # Key is the id
    #         if dialogue_acts[key].adjacency_dict is not None:
    #             for adjacency_tag in dialogue_acts[key].adjacency_dict.keys():
    #                 (letter, number, no_of_plusses) = dialogue_acts[key].adjacency_dict[adjacency_tag]
    #                 tupl = copy.deepcopy((key, letter, number, no_of_plusses))
    #                 if adjacency_tag in pairs.keys():
    #                     pairs[adjacency_tag].append(tupl)
    #                 else:
    #                     pairs[adjacency_tag] = [tupl]
    # print(pairs)





if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # parser.add_argument('--save_interval', type=int, default=500,
    #                     help='save every SAVE_INTERVAL iterations')
    # parser.add_argument('--interpolate', action='store_true', 
    #                     help = 'Include this flag to plot interpolation between two generated digits.')
    main()