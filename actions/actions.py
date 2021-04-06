from typing import Dict, Text, Any, List, Union, Optional
from rasa_sdk import Action, Tracker ,FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from joblib import load
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process
from sklearn.neighbors import KNeighborsClassifier

class ValidateElicitationForm(FormValidationAction):
    """Validating our form input using
    multiple choice answers from Harvard
    Dialect Study"""

    def name(self) -> Text:
        """Unique identifier of the form"""

        return "validate_elicitation_form"
               
    # validate user answers
    @staticmethod
    def answers_db() -> Dict[str, List]:
        """Database of supported answers."""

        return {'second_person_plural': ['you all',
                                         'yous',
                                         'youse',
                                         'you lot',
                                         'you guys',
                                         'you ’uns',
                                         'yinz',
                                         'you',
                                         'other',
                                         'y’all'],
                'bug': ['pill bug',
                        'doodle bug',
                        'potato bug',
                        'roly poly',
                        'sow bug',
                        'basketball bug',
                        'twiddle bug',
                        'roll-up bug',
                        'wood louse',
                        'millipede',
                        'centipede',
                        'I know what this creature is, but have no word for it',
                        'I have no idea what this creature is'],
                'highway': ['highway',
                            'freeway',
                            'parkway',
                            'turnpike',
                            'expressway',
                            'throughway/thru-way',
                            'a freeway is bigger than a highway',
                            'a freeway is free (i.e., doesn’t charge tolls)',
                            'a highway isn’t',
                            'a freeway has limited access (no stop lights, no intersections), whereas a highway can have stop lights and intersections'],
                'firefly': ['lightning bug',
                            'firefly',
                            'I use lightning bug and firefly interchangeably',
                            'peenie wallie',
                            'I have no word for this',
                            'other'],
                'wild_cat': ['mountain lion',
                              'cougar',
                              'puma',
                              'mountain cat ',
                              'panther',
                              'catamount',
                              'mountain screamer',
                              'painter'],
                'shoes': ['sneakersn',
                          'shoes',
                          'gym shoes',
                          'sand shoes',
                          'jumpers',
                          'tennis shoes',
                          'running shoes',
                          'runners',
                          'trainers',
                          'I have no general word for this'],
                'yard_sale': ['tag sale',
                              'yard sale',
                              'garage sale',
                              'rummage sale',
                              'thrift sale',
                              'stoop sale',
                              'carport sale',
                              'sidewalk sale',
                              'jumble ',
                              ' jumble sale',
                              'car boot ',
                              ' car boot sale',
                              'patio sale'],
                'mary_merry_marry': ['all three are pronounced the same',
                                     'all three are pronounced differently',
                                     'Mary and merry are pronounced the same, but marry is different',
                                     'merry and marry are pronounced the same, but Mary is different',
                                     'Mary and marry are pronounced the same, but merry is different'],
                'frosting': ['frosting',
                             'icing',
                             'frosting and icing refer to different things',
                             'both',
                             'neither'],
                'rubbernecking': ['rubberneck',
                                  'rubbernecking',
                                  'rubbernecking is the activity (slowing down and gawking) that causes the traffic jam, but I have no word for the traffic jam itself',
                                  'gapers’ block',
                                  'gapers’ delay',
                                  'Lookie Lou',
                                  'curiosity delay',
                                  'gawk block',
                                  'I have no word for this'],
                'cot_caught': ['different', 'same'],
                'school_college': ['gut',
                                   'crypt course',
                                   'crip course',
                                   'bird',
                                   'blow-off',
                                   'meat'],
                'freight': ['semi',
                            'semi-truck',
                            'tractor-trailer',
                            'trailer truck',
                            'transfer truck',
                            'transport',
                            'truck and trailer',
                            'semi-trailer',
                            '18-wheeler',
                            'truck',
                            'rig',
                            'big rig',
                            'lorry'],
                'second_syllabe': ['with the vowel in jam',
                                   'with the vowel in palm',
                                   'other'],
                'beverage': ['soda',
                              'pop',
                              'coke',
                              'tonic',
                              'soft drink',
                              'lemonade',
                              'cocola',
                              'fizzy drink',
                              'dope',
                              'other'],
                'sandwich': ['sub',
                             'grinder',
                             'hoagie',
                             'hero',
                             'poor boy',
                             'bomber',
                             'Italian sandwich',
                             'baguette',
                             'sarney',
                             'I have no word for this',
                             'other'],
                'brew_thru': ['brew thru',
                              'party barn',
                              'bootlegger',
                              'beer barn',
                              'beverage barn',
                              'we have these in my area, but we have no special term for them',
                              'I have never heard of such a thing',
                              'other'],
                'crawfish': ['crawfish',
                             'crayfish',
                             'craw',
                             'crowfish',
                             'crawdad',
                             'mudbug',
                             'I have no word for this critter',
                             'other'],
                'rain_sun': ['sunshower',
                             'the wolf is giving birth',
                             'the devil is beating his wife',
                             'monkey’s wedding',
                             'fox’s wedding',
                             'pineapple rain',
                             'liquid sun',
                             'I have no term or expression for this',
                             'other'],
                'road_meet_in_circle': ['rotary',
                                        'roundabout',
                                        'circle',
                                        'traffic circle',
                                        'traffic circus',
                                        'I have no word for this',
                                        'other'],
                'halloween': ['gate night',
                              'trick night',
                              'mischief night',
                              'cabbage night',
                              'goosy night',
                              'devil’s night',
                              'devil’s eve',
                              'I have no word for this',
                              'other'],
                'water_fountain': ['bubbler',
                                   'water bubbler',
                                   'drinking fountain',
                                   'water fountain',
                                   'other']
                }

    def create_validation_function(name_of_slot):
        """Function generate our validation functions, since
        they're pretty much the same for each slot"""

        def validate_slot(
            self,
            value: Text,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
        ) -> Dict[Text, Any]:
            """Validate user input."""

            if value.lower() in self.answers_db()[name_of_slot]:
                # validation succeeded, set the value of the slot to
                # user-provided value
                return {name_of_slot: value}
            else:
                # find the closest answer by some measure (edit distance?)
                choices = self.answers_db()[name_of_slot]
                answer = process.extractOne(value.lower(), choices)
                #answer=("Dallas Cowboys", 90)
                # check to see if distnace is greater than some threshold
                if answer[1] < 45:
                    # if so, set slot to "other"
                    return {name_of_slot: "other"}
                else:
                    return {name_of_slot: answer[0]}

        return(validate_slot)
    # create validation function for each question
    validate_bug = create_validation_function(name_of_slot="bug")
    validate_beverage = create_validation_function(name_of_slot="beverage")
    validate_second_person_plural = create_validation_function(
        name_of_slot="second_person_plural")
    validate_mary_merry_marry = create_validation_function(
        name_of_slot="mary_merry_marry")
    validate_cot_caught = create_validation_function(name_of_slot="cot_caught")
    validate_yard_sale = create_validation_function(name_of_slot="yard_sale")
    validate_halloween = create_validation_function(name_of_slot="halloween")
    validate_sandwich = create_validation_function(name_of_slot="sandwich")
    validate_road_meet_in_circle = create_validation_function(
        name_of_slot="road_meet_in_circle")
    validate_shoes = create_validation_function(name_of_slot="shoes")
    validate_highway = create_validation_function(name_of_slot="highway")
    validate_rain_sun = create_validation_function(name_of_slot="rain_sun")
    validate_rubbernecking = create_validation_function(
        name_of_slot="rubbernecking")
    validate_frosting = create_validation_function(name_of_slot="frosting")
    validate_firefly = create_validation_function(name_of_slot="firefly")
    validate_brew_thru = create_validation_function(name_of_slot="brew_thru")
    validate_water_fountain = create_validation_function(
        name_of_slot="water_fountain")
    validate_school_college = create_validation_function(
        name_of_slot="school_college")
    validate_freight = create_validation_function(name_of_slot="freight")
    validate_second_syllabe = create_validation_function(
        name_of_slot="second_syllabe")
    validate_wild_cat = create_validation_function(name_of_slot="wild_cat")
    validate_crawfish = create_validation_function(name_of_slot="crawfish")

class DetectDialect(Action):
    """Detect the users dialect"""

    def name(self) -> Text:
        """Unique identifier of the form"""

        return "detect_dialect"
    @staticmethod
    def slot_key_db() -> Dict[str, List]:
        """Database of slot values and corresponding questions"""

        return {'Q01':'second_person_plural',
                'Q02':'bug',
                'Q03':'highway',
                'Q04':'firefly',
                'Q05':'wild_cat',
                'Q06':'shoes',
                'Q07':'yard_sale',
                'Q08':'mary_merry_marry',
                'Q09':'frosting',
                'Q10':'highway',
                'Q11':'rubbernecking',
                'Q12':'cot_caught',
                'Q13':'school_college',
                'Q14':'freight',
                'Q15':'second_syllabe',
                'Q16':'beverage',
                'Q17':'sandwich',
                'Q18':'brew_thru',
                'Q19':'crawfish',
                'Q20':'rain_sun',
                'Q21':'road_meet_in_circle',
                'Q22':'halloween',
                'Q23':'water_fountain',
                'Q24':'firefly'} 
    def run(self, dispatcher, tracker, domain):
        """get dialect classification """
        # let user know the analysis is running
        # dispatcher.utter_message(template="utter_working_on_it")

        # get information from the form & format it
        # for encoding
        # get information from the form and fromat it
        # for encoding
        slot_question_key= self.slot_key_db()
        formatted_responses=pd.Series(index=slot_question_key.keys())
        for index,value in formatted_responses.items():
            formatted_responses[index]=tracker.get_slot(slot_question_key[index])
        # classify test case
        # TODO: use user input instead of test case
        dialects = ClassifierPipeline_knn().get_top_3_knn(formatted_responses)
        state_1, state_2, state_3 = dialects
        dialects = f"The state that mostly closely matches your language use is {state_1}, followed by {state_2} and {state_3}"

        # always guess Test cities for now
        return [SlotSet("dialect", dialects)]
class ClassifierPipeline_knn:
    """Load in classifier & encoders"""

    def name(self) -> Text:
        """Unique identifier of the classfier """

        return "5knn_state"

    def encode_answers(self, input_data):
        """Reads in the sample encoded data w/ correct columns and
        converts input data to the same format"""
        # read in empty dataframe with correct columns
        encoding_sample = pd.read_csv("actions\\empty_data_structure.csv")

        # transpose input data
        input_data = pd.DataFrame(input_data).transpose()

        # one hot encode input data & standardize column names
        encoded_input_data = pd.get_dummies(input_data)
        encoded_input_data.columns = [
            col.replace(" ", "_") for col in encoded_input_data.columns
        ]

        # encode it
        encoded_data = encoding_sample.align(encoded_input_data, join="left", axis=1)

        # convert na's to 0 (since we're one hot encoding)
        encoded_data = encoded_data[1].fillna(0)

        return encoded_data

    def get_top_3_knn(self, data):
        """Read in the knn model and apply it to correctly formatted sample data"""
        # read in model
        state_knn = load("actions\\state_level_knn.joblib.dat")

        # encode input data
        encoded_data = self.encode_answers(input_data=data)

        pred = state_knn.predict_proba(encoded_data)
        top_3 = np.argsort(pred, axis=1)[:, -3:]
        results = [state_knn.classes_[i] for i in top_3]

        return results[0].tolist()
