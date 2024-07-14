import json
import re

from spacy import displacy
from spacy.tokens import Doc, Span


def preprocess_text(text):
    # This regex replaces sequences of three or more dots with a single '<UNK>'
    text = re.sub(r'\.{3,}', '{UNK}', text)
    text = re.sub(r'\[', '', text)
    text = re.sub(r']', '', text)
    return text


def load_data(xai=False, filename='outputs/output_hg_lora.txt'):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        while True:
            # Read the original sentence
            sentence = file.readline().strip()
            if xai:
                akkad_sentence = file.readline().strip()
                entities_text = file.readline().strip()
            if not sentence:  # End of file check
                break

            # Initialize a variable to collect the LLM output
            llm_output = ''

            # Read until you encounter a pipe '|'
            while True:
                line = file.readline().strip()
                if '|' in line:  # Check if the line contains a pipe
                    llm_output += line[:line.index('|')]  # Add only up to the pipe
                    break
                llm_output += line

            if xai:
                json_matches = re.findall(r'\{.*?\}', llm_output)
                entities = []
                for json_match in json_matches:
                    try:
                        entity = json.loads(json_match.replace('\'', '\"'))
                        if "Entity English Text" in entity.keys():
                            entities.append(entity)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON: {json_match}")

            else:
                # Extract JSON from the LLM output using regex
                json_match = re.search(r'\{.*\}', llm_output)
                if json_match:
                    json_string = json_match.group()
                    try:
                        entities = json.loads(json_string)
                        data.append((sentence, entities))
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON for sentence: {sentence}")

    return data


def highlight_text(sentence, entities, lang):
    # Sort entities by start index to ensure correct insertion order
    start_key, end_key = (f'start_{lang}', f'end_{lang}') if lang else ('start', 'end')
    entities = sorted(entities, key=lambda x: x[start_key], reverse=True)

    # Adjust the sentence to incorporate highlights
    highlighted_text = sentence
    for entity in entities:
        start, end = entity[start_key], entity[end_key]
        # Embed the entity text with a span that includes a style for background color
        entity_html = f"<span style='background-color:{get_color_for_type(normalize_entity_type(entity['type']))};'>{highlighted_text[start:end]}</span>"
        highlighted_text = highlighted_text[:start] + entity_html + highlighted_text[end:]

    return highlighted_text


def normalize_entity_type(entity_type):
    # Normalize different entity types based on explicit checks
    entity_type = entity_type.lower()

    if 'commodit' in entity_type:
        return 'Commodities'
    if 'location' in entity_type:
        return 'Locations'
    if 'document' in entity_type:
        return 'Documents'
    if 'animal' in entity_type:
        return 'Animals'
    if 'person' in entity_type:
        return 'Person'
    if 'judge' in entity_type:
        return 'Judge'
    if 'adoptee' in entity_type:
        return 'Adoptee'
    if 'adopter' in entity_type:
        return 'Adopter'
    if 'parent' in entity_type:
        return 'Parent'
    if 'apprentice' in entity_type:
        return 'Apprentice'
    if 'apprentice giver' in entity_type:
        return 'Apprentice Giver'
    if 'trainer' in entity_type:
        return 'Trainer'
    if 'obligee' in entity_type:
        return 'Obligee'
    if 'obligor' in entity_type:
        return 'Obligor'
    if 'litigant' in entity_type:
        return 'Litigant'
    if 'defendant' in entity_type:
        return 'Defendant'
    if 'plaintiff' in entity_type:
        return 'Plaintiff'
    if 'deponent' in entity_type:
        return 'Deponent'
    if 'heir' in entity_type:
        return 'Heir'
    if 'testator' in entity_type:
        return 'Testator'
    if 'exchanger' in entity_type:
        return 'Exchanger'
    if 'donee' in entity_type:
        return 'Donee'
    if 'donor' in entity_type:
        return 'Donor'
    if 'lessee' in entity_type:
        return 'Lessee'
    if 'lessor' in entity_type:
        return 'Lessor'
    if 'creditor' in entity_type:
        return 'Creditor'
    if 'debtor' in entity_type:
        return 'Debtor'
    if 'manumitter' in entity_type:
        return 'Manumitter'
    if 'manumitted slave' in entity_type:
        return 'Manumitted Slave'
    if 'bride' in entity_type:
        return 'Bride'
    if "bride's agent" in entity_type or "bride’s agent" in entity_type:
        return "Bride's Agent"
    if 'groom' in entity_type:
        return 'Groom'
    if "groom's Agent" in entity_type or "groom’s agent" in entity_type:
        return "Groom's Agent"
    if 'king' in entity_type:
        return 'King'
    if 'oath taker' in entity_type:
        return 'Oath Taker'
    if 'business partner' in entity_type:
        return 'Business Partner'
    if 'prebend-giver' in entity_type:
        return 'Prebend-giver'
    if 'prebend-holder' in entity_type:
        return 'Prebend-holder'
    if 'prebend-performer' in entity_type:
        return 'Prebend-performer'
    if 'buyer' in entity_type:
        return 'Buyer'
    if 'seller' in entity_type:
        return 'Seller'
    if 'payer' in entity_type:
        return 'Payer'
    if 'recipient' in entity_type:
        return 'Recipient'
    if 'summoner' in entity_type:
        return 'Summoner'
    if 'person to be summoned' in entity_type:
        return 'Person to be summoned'
    if 'guarantor' in entity_type:
        return 'Guarantor'
    if 'witness' in entity_type:
        return 'Witness'
    if 'scribe' in entity_type:
        return 'Scribe'
    # Return the original entity type if no normalization rule applies
    return entity_type.capitalize()

def get_color_for_type(entity_type):
    # Define colors for each entity type
    colors = {
        'Commodities': '#7FFFD4',
        'Locations': '#00FFFF',
        'Documents': '#4682B4',
        'Animals': '#A52A2A',
        'Adoptee': '#DEB887',
        'Adopter': '#5F9EA0',
        'Parent': '#D2691E',
        'Apprentice': '#FF7F50',
        'Apprentice Giver': '#6495ED',
        'Trainer': '#FFF8DC',
        'Obligee': '#DC143C',
        'Obligor': '#00FFFF',
        'Judge': '#00008B',
        'Litigant': '#008B8B',
        'Defendant': '#B8860B',
        'Plaintiff': '#A9A9A9',
        'Deponent': '#006400',
        'Heir': '#BDB76B',
        'Testator': '#8B008B',
        'Exchanger': '#556B2F',
        'Donee': '#FF8C00',
        'Donor': '#9932CC',
        'Lessee': '#8B0000',
        'Lessor': '#E9967A',
        'Creditor': '#8FBC8F',
        'Debtor': '#483D8B',
        'Manumitter': '#2F4F4F',
        'Manumitted Slave': '#00CED1',
        'Bride': '#9400D3',
        'Bride’s Agent': '#FF1493',
        'Groom': '#00BFFF',
        'Groom’s Agent': '#696969',
        'King': '#1E90FF',
        'Oath Taker': '#B22222',
        'Business Partner': '#FFFAF0',
        'Prebend-giver': '#228B22',
        'Prebend-holder': '#FF00FF',
        'Prebend-performer': '#DCDCDC',
        'Buyer': '#F8F8FF',
        'Seller': '#FFD700',
        'Payer': '#DAA520',
        'Recipient': '#808080',
        'Summoner': '#008000',
        'Person to be summoned': '#ADFF2F',
        'Guarantor': '#F0FFF0',
        'Witness': '#FF69B4',
        'Scribe': '#CD5C5C'
    }
    # Default to a grey color if the entity type is not found
    return colors.get(entity_type, '#D3D3D3')


def render_entities(doc):
    # Render entities using displacy
    options = {"colors": {"Commodity": "green",
                          "Location": "SlateBlue",
                          "Document": "lightblue",
                          "Animal": "orange",
                          "Adoptee": "red",
                          "Adopter": "darkred",
                          "Parent": "yellow",
                          "Apprentice": "teal",
                          "Apprentice Giver": "lightgreen",
                          "Trainer": "brown",
                          "Obligee": "pink",
                          "Obligor": "lightcoral",
                          "Judge": "gold",
                          "Litigant": "lightyellow",
                          "Defendant": "salmon",
                          "Plaintiff": "magenta",
                          "Deponent": "cyan",
                          "Heir": "darkblue",
                          "Testator": "darkgreen",
                          "Exchanger": "darkorange",
                          "Donee": "lightpink",
                          "Donor": "lightgray",
                          "Lessee": "darkgray",
                          "Lessor": "navy",
                          "Creditor": "crimson",
                          "Debtor": "khaki",
                          "Manumitter": "lightcyan",
                          "Manumitted Slave": "peachpuff",
                          "Bride": "lavender",
                          "Bride’s Agent": "coral",
                          "Groom": "darkkhaki",
                          "Groom’s Agent": "lightseagreen",
                          "King": "maroon",
                          "Oath Taker": "olive",
                          "Business Partner": "silver",
                          "Prebend-giver": "wheat",
                          "Prebend-holder": "lime",
                          "Prebend-performer": "turquoise",
                          "Buyer": "peru",
                          "Seller": "violet",
                          "Payer": "orchid",
                          "Recipient": "DarkSalmon",
                          "Summoner": "plum",
                          "Person to be summoned": "bisque",
                          "Guarantor": "thistle",
                          "Witness": "mintcream",
                          "Scribe": "darkslategray"}}
    html = displacy.render(doc, style="span", jupyter=False, options=options)
    return html


def convert_char_to_token_indices(sentence, entities, nlp, lang=None):
    doc = nlp(sentence)
    token_indices = []
    start_key, end_key = (f'start_{lang}', f'end_{lang}') if lang else ('start', 'end')

    for ent in entities:
        char_start = ent[start_key]
        char_end = ent[end_key]

        # Find the token indices corresponding to the character indices
        token_start = None
        token_end = None

        for token in doc:
            if token.idx == char_start:
                token_start = token.i
            if token.idx + len(token) == char_end:
                token_end = token.i + 1

        if token_start is not None and token_end is not None:
            token_indices.append({
                'start': token_start,
                'end': token_end,
                'label': ent['type']
            })
        else:
            print(f"Warning: Could not find token indices for entity {ent}")

    return token_indices


def create_custom_doc(nlp, text, entities):
    # Tokenize the text to create a Doc object
    doc = nlp(text)

    # Create a list of spans with custom entities
    spans = [Span(doc, ent['start'], ent['end'], label=ent['label']) for ent in entities]

    # Assign the spans to the doc.ents
    doc.spans["sc"] = spans
    return doc

