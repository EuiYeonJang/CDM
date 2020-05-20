import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle
import copy
from tqdm import tqdm
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk import download as nltk_download
import copy
nltk_download('wordnet')
""" Here, we process the AMI files into data structures usable for our analysis.


"""

"""
Buildup of AMI corpus:
- AjacencyPairs - each refers to a source(a) and target(b) (not always both) dialogue act, and their types
- DialogueActs - refers to words in href. Speaker is defined by capital in filename
- CorpusResources/meetings.xml defines the speakers for each meeting. global_name is speaker id, and starts with gender

"""


class Word():
    def __init__(self):
        self.id = ''
        self.text = ''
        self.start_time = ''
        self.end_time = ''
        self.meeting_id = ''
        self.participant_letter = ''

    def __init__(self, id, text, start_time, end_time, meeting_id, participant_letter):
        self.id = id
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.meeting_id = meeting_id
        self.participant_letter = participant_letter


class Speaker():
    def __init__(self, tag, gender, age, education):
        self.tag = tag # This is the speaker id
        self.gender = gender
        self.age = age
        self.education = education

class DialogueAct():
    # TODO: Include calling of text and participant functions in initialisation
    def __init__(self, id, meeting_id, start_time, end_time, participant_id, type_, original_type, channel, 
    comment, adjacency, start_word_id, end_word_id, words, speakers):
        self.id = id
        self.meeting_id = meeting_id
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
    
    # def retrieve_text(self, words_dict):
    #     # retrieve index in words_dict of start_word_id
    #     start_word_index = list(words_dict.keys()).index(self.start_word_id)
    #     if self.end_word_id is not None:
    #         end_word_index = list(words_dict.keys()).index(self.end_word_id)+1
    #     else:
    #         end_word_index = start_word_index+1

    #     act_words = list(words_dict.values())[start_word_index:end_word_index]
    #     string = ' '.join([word.text for word in act_words if word.text])
    #     # self.text = string
    #     print(string)
    #     return string
    
    def retrieve_participant(self, speakers):
        participant_id = self.participant_id
        participant = speakers[participant_id]
        print(participant.tag)
        return participant

    # def list_adjacency(self):
    # # also call a function to split the adjacency tags into (1) a list with all tags, and (2) a list of the number of plusses
    # # Or we make it a dict! tag: (a/b, number_of_plusses, dash_number)
    # # A dash indicates a split (so multiple equal utterances in the adjacency pair, by different speakers)
    # # A plus indicates another utterance with the same function, by one of the same speakers. (So if the replier utters two 
    # # utterances in a row, the second gets a plus)
    # # Then use these tags to create a set of all adjacency tags.
    #     # print(self.adjacency)
    #     if self.adjacency is not None:
    #         adjacencies = self.adjacency.split('.')
    #         print(adjacencies)
    #         # There is also a dash - 
    #         adjacency_dict = {}
    #         for tag in adjacencies:
    #             print("Tag: ", tag)
    #             if (tag.count('a') + tag.count('b')) != 1:
    #                 continue
    #             # Select the first a or b letter, this ends the adjacency pair tag
    #             letter_index = max(tag.find('a'), tag.find('b')) # Removes -1 from unfound letter
    #             final_tag = tag[:letter_index]
    #             letter = tag[letter_index]
    #             number = 1
    #             print("final tag: ", final_tag, letter_index)
    #             if '-' in tag:
    #                 number = tag[letter_index+2]
                
    #             no_of_plusses = tag.count('+')

    #             adjacency_dict[final_tag] = (letter, number, no_of_plusses)

    #         return adjacency_dict

    #     else:
    #         return None

def unpickle_or_generate(gen_fun, pickle_path, *args):
    if not os.path.isfile(pickle_path):
        obj = gen_fun(*args)
        with open(pickle_path, 'wb') as file:
            pickle.dump(obj, file)
    else:
        with open(pickle_path, 'rb') as file:
            obj = pickle.load(file)
    return obj

def read_string_until(string, end_before, return_rest = False):
    end_index = string.index(end_before)
    if return_rest:
        return string[:end_index], string[1+end_index:]
    else:
        return string[:end_index]

def meeting_id_and_letter(filename):
    meeting_id, rest = read_string_until(filename, '.', return_rest=True)
    participant_letter = read_string_until(rest, '.')
    # print(filename, meeting_id, rest, participant_letter)
    return meeting_id, participant_letter


def extract_words(words_directory):
    # Note: store the meeting id and participant letter in this as well 
    # Meeting id example: ES2002a ( out of EN2001a.A.words.xml )
    # Participant letter example: A ( out of EN2001a.A.words.xml)
    wnl = WordNetLemmatizer()
    words = {}
    words_by_id = {}
    meeting_id = ''
    participant_letter = ''
    # structure: words = {meeting_id: word_id: Word}
    for subdir, dirs, files in os.walk(words_directory):
        for file in sorted(files): # Sorted to ensure files are read in alphabetic order. TODO: This is an easy fix. Could make it safer by checking if key exists
            filepath = subdir + os.sep + file

            if filepath.endswith('.xml'):
                tree = ET.parse(filepath)
                root = tree.getroot()
                
                old_meeting_id = copy.deepcopy(meeting_id)
                old_participant_letter = copy.deepcopy(participant_letter)
                meeting_id, participant_letter = meeting_id_and_letter(file)
                if meeting_id != old_meeting_id: # Apparently, ES2015b.A gets read out many steps before ES2015b.B
                    words[old_meeting_id] = words_by_id
                    # if old_meeting_id == 'ES2015b' and old_participant_letter == 'A':
                    #     print(old_meeting_id, participant_letter)
                    #     print("Found it!", [word.id for word in list(words_by_id.values())[0:10]])
                    #     print("Found it!", [word for word in list(words_by_id.keys())[0:10]])
                    words_by_id = {}
                    

                for child in root:
                    id = child.attrib['{http://nite.sourceforge.net/}id']
                    start_time = child.attrib.get('starttime', None)
                    end_time = child.attrib.get('endtime', None)
                    if child.tag == 'w' and child.text.isalnum():
                        text = wnl.lemmatize(child.text.lower())
                    else:
                        text = False
                            # if child.tag == 'vocalsound' and child.attrib['type'] == 'laugh':
                            #     print(child.id)
                    words_by_id[id] = Word(id, text, start_time, end_time, meeting_id, participant_letter)
                    # else:
                    #     id = child.attrib['{http://nite.sourceforge.net/}id']
                    #     start_time = child.attrib.get('starttime', None)
                    #     end_time = child.attrib.get('endtime', None)
                    #     c = child.attrib.get('c', None)
                    #     t = child.attrib.get('t', None)
                    #     k = child.attrib.get('k', None)
                    #     qut = child.attrib.get('qut', None)
                    #     text = False
                    #     words[id] = Word(id, text, start_time, end_time, c, t = t, k = k, qut = qut)
    words[meeting_id] = words_by_id # Store the last one as well

    return words

def extract_speakers(speakerspath, meetings_path):
    print("Extracting speakers file..")

    tree = ET.parse(meetings_path)
    root = tree.getroot()
    speakers = {}
    # Structure: {meeting_id: participant_letter: gender}
    for meeting in tqdm(root):
        tag = meeting.attrib['observation']
        speakers_in_meeting = {} 
        for speaker in meeting:
            speaker_tag = speaker.attrib['nxt_agent']
            gender = speaker.attrib['global_name'][0].lower()
            # print(gender)
            speakers_in_meeting[speaker_tag] = gender
        speakers[tag] = speakers_in_meeting
    return speakers
    #     gender = child.attrib.get('gender', None)
    #     for grandchild in child:
    #         if grandchild.tag == 'age':
    #             age = grandchild.text
    #         elif grandchild.tag == 'education':
    #             education = grandchild.text
    #     speakers[tag] = Speaker(tag, gender, age, education)
    # return speakers


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

def dialogue_acts_words(href, words): # Retrieves words given the href from a dialogue act
    #TODO: Check in ICSI if the in-between words not only use words from a single speaker! 
    # Words files are split between speakers, so if we just use the words in a certain range, this will fuck up
    opening_bracket_index = href.find('(')
    word_ids = href[opening_bracket_index+1:]
    first_closing_bracket_index = word_ids.find(')')

    meeting_id, participant_letter = meeting_id_and_letter(href)

    start_word_id = word_ids[:first_closing_bracket_index]
    end_word_id = word_ids[first_closing_bracket_index:]
    end_opening_bracket = end_word_id.find('(')
    if end_opening_bracket != -1:
        end_word_id = end_word_id[end_opening_bracket+1:-1]
    else:
        end_word_id = None
    
    start_word_index = list(words[meeting_id].keys()).index(start_word_id)
    if end_word_id is not None:
        end_word_index = list(words[meeting_id].keys()).index(end_word_id)+1
    else:
        end_word_index = start_word_index+1
    act_words = list(words[meeting_id].values())[start_word_index:end_word_index]
    # print('\n \n')
    # print("Hello: ", act_words[0].start_time)
    words_list = [word.text for word in act_words if word.text != False]
    start_time = act_words[0].start_time
    end_time = act_words[-1].end_time
    counter = Counter(words_list)
    # print(counter)
    return counter, start_time, end_time

def adjacency_act_id(href):
    opening_bracket_index = href.find('(') # only brackets around the id
    id = href[opening_bracket_index+1:-1] # last character is closing bracket
    return id

def extract_dialogue_acts(acts_directory, words, speakers): #TODO: Note: About 30 of the 170 (sub)meetings do not have dialogue act/adjacency pair annotation
    print("Extracting dialogue acts..")

    # Create a dict that contains dialogue_act_id: (Counter of words, meeting_id, participant_letter, starting_time)
    # Afterwards, we can convert meeting_id and participant_letter to f or m, by using participants.xml
    # And then create a,b pairs using adjacency_pairs files
    dialogue_acts = {}
    dialogue_acts_by_id = {}
    meeting_id = ''
    for subdir, dirs, files in tqdm(os.walk(acts_directory)):
        for file in tqdm(sorted(files)):
            filepath = subdir + os.sep + file
            if filepath.endswith('dialog-act.xml'):
                # old_meeting_id = copy.deepcopy(meeting_id)
                
                old_meeting_id = copy.deepcopy(meeting_id)
                meeting_id, participant_letter = meeting_id_and_letter(file)
                # print("ID: ", meeting_id,"Letter: ", participant_letter)
                if meeting_id != old_meeting_id:
                    dialogue_acts[old_meeting_id] = dialogue_acts_by_id
                    dialogue_acts_by_id = {}  
                #     if len(meeting_dict_time) > 0:
                #         dialogue_acts_time[meeting_id] = meeting_dict_time
                #         dialogue_acts_text[meeting_id] = meeting_dict_text
                #         print(meeting_dict_time)
                #         print(meeting_dict_text)
                #     meeting_dict_time = {}
                #     meeting_dict_text = {}
                tree = ET.parse(filepath)
                root = tree.getroot()
                for child in root:

                    id = child.attrib['{http://nite.sourceforge.net/}id']
                    # start_time = child.attrib.get('starttime', None)
                    # adjacency = child.attrib.get('adjacency', None)
                    # participant_id = child.attrib.get('participant', None)

                    for grandchild in child:
                        # print(grandchild.tag)
                        if grandchild.tag == '{http://nite.sourceforge.net/}child': # Assume each dialogue act has only one 'child' child
                            href = grandchild.attrib.get('href', None)
                            counter, start_time, end_time = dialogue_acts_words(href, words)
                    gender = speakers[meeting_id][participant_letter]
                    dialogue_acts_by_id[id] = (counter, meeting_id, gender, start_time, end_time)
    dialogue_acts[meeting_id] = dialogue_acts_by_id # Also store the last one
    return dialogue_acts
   
def retrieve_between_counter(dialogue_acts, meeting_id, start_time, end_time):
    acts_in_meeting = dialogue_acts[meeting_id]

    acts_between_male = sum([act[0] for act in acts_in_meeting 
                                if (act[3] > start_time and act[4] < end_time and act[2] == 'm')], 
                                start = Counter() )
    acts_between_female = sum([act[0] for act in acts_in_meeting 
                                if (act[3] > start_time and act[4] < end_time and act[2] == 'f')], 
                                start = Counter() )
    return acts_between_male, acts_between_female

def extract_adjacency_pairs(acts_directory, words, speakers, dialogue_acts):
    print("Extracting adjacency pairs...")
    adjacency_pairs = {}

    f_m = []
    m_m = []
    m_f = []
    f_f = []

    missing_turns = 0
    total_pairs = 0
    for subdir, dirs, files in tqdm(os.walk(acts_directory)):
        for file in tqdm(files):
            filepath = subdir + os.sep + file

            if filepath.endswith('adjacency-pairs.xml'):
                meeting_id, participant_letter = meeting_id_and_letter(file)

                tree = ET.parse(filepath)
                root = tree.getroot()
                for child in root:

                    id = child.attrib['{http://nite.sourceforge.net/}id']

                    adjacency_pair = {}
                    for grandchild in child: 
                        role = grandchild.attrib['role']
                        # print("Role: ", role)
                        if role != 'source' and role != 'target':
                            # print("Not a valid source!")
                            continue
                        href = grandchild.attrib.get('href', None)
                        dialogue_act_id = adjacency_act_id(href)

                        tag = ('a' if role == 'source' else 'b')
                        # print("Tag: ", tag, adjacency_pair)
                        dialogue_act = dialogue_acts[meeting_id][dialogue_act_id]
                        # print("D: ", dialogue_act)
                        gender = dialogue_act[2] #TODO: Dialogue acts now include gender
                        start_time = dialogue_act[3]
                        end_time = dialogue_act[4]
                        counter = dialogue_act[0]
                        adjacency_pair[tag] = (counter, gender, start_time, end_time)
                    # tqdm.write(f"Pair: {adjacency_pair}")

                    # Now, store all in-between words of this adjacency pair
                    total_pairs += 1
                    if not ('a' in adjacency_pair.keys() and 'b' in adjacency_pair.keys()):
                        missing_turns += 1
                        continue
                    start_time = adjacency_pair['a'][3] # end time of first utterance
                    end_time = adjacency_pair['b'][2] #start time of last utterance

                    adjacency_pair['mb'], adjacency_pair['fb'] = retrieve_between_counter(dialogue_acts, meeting_id, start_time, end_time)
                    
                    new_pair = {'a': adjacency_pair['a'][0], 'b': adjacency_pair['b'][0], 
                        'mb': adjacency_pair['mb'], 'fb': adjacency_pair['fb']}

                    if adjacency_pair['a'][1] == 'm' and adjacency_pair['b'][1] == 'm':
                        m_m.append(new_pair)
                    elif adjacency_pair['a'][1] == 'm' and adjacency_pair['b'][1] == 'f':
                        f_m.append(new_pair)
                    elif adjacency_pair['a'][1] == 'f' and adjacency_pair['b'][1] == 'm':
                        m_f.append(new_pair)
                    elif adjacency_pair['a'][1] == 'f' and adjacency_pair['b'][1] == 'f':
                        f_f.append(new_pair)
    print("Number of missing turns: ", missing_turns, total_pairs) #TODO: Investigate this
    return (f_m, m_m, m_f, f_f)

# def process_adjacency_pairs(adjacency_dict):

#     adjacency_pairs = []
#     # Now we sould convert the 'text' entries to counters, and separate each a-b pair.
#     for AP_tag in adjacency_dict.keys():
#         print(AP_tag, adjacency_dict[AP_tag])
#         # this contains a list of dialogue acts that belong to this one adjacency pair
#         for dialogue_act in adjacency_dict[AP_tag].get('a', []):
#             a_counter = Counter()
#             text = dialogue_act[2]
#             # a_starttime = dialogue_act[4]
#             for utterance in text:
#                 list_of_tokens = utterance
#                 a_counter.update(list_of_tokens)

#             a_participant_gender = dialogue_act[3][0]
#             for dialogue_act in adjacency_dict[AP_tag].get('b', []):
#                 print("Found b!")
#                 b_counter = Counter()
#                 text = dialogue_act[2]
#                 # b_starttime = dialogue_act[4]
#                 for utterance in text:
#                     list_of_tokens = utterance #.lower() #.split(' ')
#                     b_counter.update(list_of_tokens)
#                 b_participant_gender = dialogue_act[3][0]
#                 dic = {
#                     'a': 
#                         {   'counter': a_counter, 
#                             'gender': a_participant_gender
#                         },
#                     'b': 
#                         {   'counter': b_counter, 
#                             'gender': b_participant_gender
#                         }   
#                         }
#                 adjacency_pairs.append(dic)
#     return adjacency_pairs

def process_inbetween(adjacency_dict, dialogue_acts_time, dialogue_acts_text):
    # Note: This can be done by using purely the words files
    adjacency_pairs = []
    # Now we sould convert the 'text' entries to counters, and separate each a-b pair.
    for AP_tag in tqdm(adjacency_dict.keys()):
        # print(AP_tag, adjacency_dict[AP_tag])
        meeting_id = AP_tag[:6]


        time_values = np.array(list(dialogue_acts_time[meeting_id].values())).astype(np.float)
        time_sorted_indices = np.argsort(time_values) # Sorts in ascending order
        text_values = list(dialogue_acts_text[meeting_id].values()) #Each value is a (text, gender) tuple, so this returns a list of those tuples
        # print(dialogue_acts_time[meeting_id].values())
        # print(time_values)
        # print(dialogue_acts_text)
        # text_sorted = list(dialogue_acts_text.values())[time_sorted_indices] 
        # print("A: ", len(time_values), len(text_values))
        time_text_dict = {time_values[index]:text_values[index] for index in time_sorted_indices}


        # this contains a list of dialogue acts that belong to this one adjacency pair
        for dialogue_act in adjacency_dict[AP_tag].get('a', []):
            a_counter = Counter()
            text = dialogue_act[2]
            a_starttime = dialogue_act[4]
            for utterance in text:
                list_of_tokens = utterance #.lower() #.split(' ')
                a_counter.update(list_of_tokens)

            a_participant_gender = dialogue_act[3][0]
            for dialogue_act in adjacency_dict[AP_tag].get('b', []):
                # print("Found b!")
                b_counter = Counter()
                text = dialogue_act[2]
                b_starttime = dialogue_act[4]
                for utterance in text:
                    list_of_tokens = utterance #.lower().split(' ')
                    b_counter.update(list_of_tokens)
                b_participant_gender = dialogue_act[3][0]

                # in_between_counter = Counter()
                # Make a male and female in between counter, which are separately counted
                male_between_counter = Counter()
                female_between_counter = Counter()
                for (time, text) in time_text_dict.items():
                    if time > float(a_starttime) and time < float(b_starttime):
                        
                        if text[1] == 'f':
                            female_between_counter.update(text[0])
                        elif text[1] == 'm':
                            male_between_counter.update(text[0])
                        else:
                            print("Gender not defined correctly.")



                dic = {
                    'a': 
                        {   'counter': a_counter, 
                            'gender': a_participant_gender
                            # 'time': a_starttime
                        },
                    'b': 
                        {   'counter': b_counter, 
                            'gender': b_participant_gender
                            # 'time': b_starttime
                        },
                    'male_between':
                        {
                            'counter': male_between_counter,
                            'gender': 'm'                                
                        },
                    'female_between':
                        {
                            'counter': female_between_counter,
                            'gender': 'f'
                        }
                        }
                adjacency_pairs.append(dic)

    return adjacency_pairs

def split_genders(adjacency_pairs):
    f_m = []
    m_m = []
    m_f = []
    f_f = []

    for adjacency_pair in adjacency_pairs:
        new_pair = {'a': adjacency_pair['a']['counter'], 'b': adjacency_pair['b']['counter']}
        if adjacency_pair['a']['gender'] == 'm' and adjacency_pair['b']['gender'] == 'm':
            m_m.append(new_pair)
        elif adjacency_pair['a']['gender'] == 'm' and adjacency_pair['b']['gender'] == 'f':
            f_m.append(new_pair)
        elif adjacency_pair['a']['gender'] == 'f' and adjacency_pair['b']['gender'] == 'm':
            m_f.append(new_pair)
        elif adjacency_pair['a']['gender'] == 'f' and adjacency_pair['b']['gender'] == 'f':
            f_f.append(new_pair)
    
    return (f_m, m_m, m_f, f_f)

def split_genders_between(in_between_pairs):
    f_m = []
    m_m = []
    m_f = []
    f_f = []

    for adjacency_pair in in_between_pairs:
        new_pair = {'a': adjacency_pair['a']['counter'], 'b': adjacency_pair['b']['counter'], 
        'mb': adjacency_pair['male_between']['counter'], 'fb': adjacency_pair['female_between']['counter']}
        if adjacency_pair['a']['gender'] == 'm' and adjacency_pair['b']['gender'] == 'm':
            m_m.append(new_pair)
        elif adjacency_pair['a']['gender'] == 'm' and adjacency_pair['b']['gender'] == 'f':
            f_m.append(new_pair)
        elif adjacency_pair['a']['gender'] == 'f' and adjacency_pair['b']['gender'] == 'm':
            m_f.append(new_pair)
        elif adjacency_pair['a']['gender'] == 'f' and adjacency_pair['b']['gender'] == 'f':
            f_f.append(new_pair)
    
    return (f_m, m_m, m_f, f_f)

def main():
    print("Extracting ICSI..")
    # TODO: Create only a single time-based dict in extract_adjacency_pairs, which has all info necessary.
    # TODO: Merge process_adjacency_pairs and process_inbetween functions
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    relative_AMI_path = '/../Corpora/AMI/'
    AMI_path = this_file_path + relative_AMI_path
    parsed_words_path = AMI_path + 'parsed_words.pkl'
    # parsed_acts_path = ICSI_path + 'parsed_acts.pkl'
    # adjacency_pair_list_pickle = ICSI_path + 'adjacency_list.pkl'
    parsed_speakers_path = AMI_path + 'parsed_speakers.pkl'
    parsed_gender_split = AMI_path + 'parsed_gender_split.pkl'
    words_directory = AMI_path + 'words/'
    # adjacency
    acts_directory = AMI_path + 'dialogueActs/'
    meetingspath = AMI_path + 'corpusResources/meetings.xml'
    speakerspath = AMI_path + 'corpusResources/participants.xml'
    parsed_acts_path = AMI_path + 'parsed_acts.pkl'
    split_path = AMI_path + 'split.pkl'
    # all_dialogue_acts_path = ICSI_path + 'speakers.xml'
    # in_between_path = ICSI_path + 'in_between.pkl'
    # split_in_between_path = ICSI_path + 'in_between_split.pkl'


    words = unpickle_or_generate(extract_words, parsed_words_path, words_directory)

    print("Number of words: ", len(words))
    # speakers = unpickle_or_generate(extract_speakers, parsed_speakers_path, speakerspath)
    # print(len(speakers))

    speakers = unpickle_or_generate(extract_speakers, parsed_speakers_path, speakerspath, meetingspath)
    print("Number of speakers: ",len(speakers))

    dialogue_acts = unpickle_or_generate(extract_dialogue_acts, parsed_acts_path, acts_directory, words, speakers)
    print("Number of dialogue acts: ",len(dialogue_acts))

    # adjacency_dict = unpickle_or_generate(extract_adjacency_pairs, parsed_adjacency_path, acts_directory, words)

    (f_m, m_m, m_f, f_f) = unpickle_or_generate(extract_adjacency_pairs, split_path, acts_directory, words, speakers, dialogue_acts)

    print("Lengths: ", len(f_m), len(m_m), len(m_f), len(f_f))

    # adjacency_pairs = unpickle_or_generate(process_adjacency_pairs, adjacency_pair_list_pickle, adjacency_dict)

    # in_between_pairs = unpickle_or_generate(process_inbetween, in_between_path, adjacency_dict, dialogue_acts_time, dialogue_acts_text)

    # (f_m, m_m, m_f, f_f) = unpickle_or_generate(split_genders, parsed_gender_split, adjacency_pairs)
    # (f_m_b, m_m_b, m_f_b, f_f_b) = unpickle_or_generate(split_genders_between, split_in_between_path, in_between_pairs)

    # print('\n \n')
    # print(len(f_m_b[2]['fb']))
    # number_of_not_in_between =  sum( [1 for pair in f_m_b if len(pair['mb']) == 0] ) + \
    #                             sum( [1 for pair in m_m_b if len(pair['mb']) == 0] ) + \
    #                             sum( [1 for pair in m_f_b if len(pair['fb']) == 0] ) + \
    #                             sum( [1 for pair in f_f_b if len(pair['fb']) == 0] ) 
    # total_number_of_pairs = len(f_m_b)+len(m_m_b) + len(m_f_b) + len(f_f_b)
    # print('Pairs with turns in between: ', total_number_of_pairs - number_of_not_in_between,'out of', total_number_of_pairs)
    # # Each of these is a list of dicts, with four keys: a, b, and mb (male between) and fb (female between)
    

    # number_of_empty_counters =  sum( [1 for pair in f_m_b if (len(pair['a']) == 0 or len(pair['b'])==0) ] )   + \
    #                             sum( [1 for pair in m_m_b if (len(pair['a']) == 0 or len(pair['b'])==0) ] )   + \
    #                             sum( [1 for pair in m_f_b if (len(pair['a']) == 0 or len(pair['b'])==0) ] )  + \
    #                             sum( [1 for pair in f_f_b if (len(pair['a']) == 0 or len(pair['b'])==0) ] )  
    # # total_number_of_pairs = len(f_m_b)+len(m_m_b) + len(m_f_b) + len(f_f_b)
    # print("Number of empty counters: ", number_of_empty_counters, 'out of', total_number_of_pairs)
    # print(f_m_b[0])
    # print(m_m_b[0])
    # print(f_f_b[0])
    # print(m_f_b[0])
    # print(adjacency_dict)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # parser.add_argument('--save_interval', type=int, default=500,
    #                     help='save every SAVE_INTERVAL iterations')
    # parser.add_argument('--interpolate', action='store_true', 
    #                     help = 'Include this flag to plot interpolation between two generated digits.')
    main()