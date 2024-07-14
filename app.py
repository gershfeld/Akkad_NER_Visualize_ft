import re

import spacy
import streamlit as st
import json
from rapidfuzz import fuzz

from utils import load_data, preprocess_text, highlight_text, convert_char_to_token_indices, create_custom_doc, \
    render_entities, get_color_for_type, normalize_entity_type, load_data_two_files


def process_data(data):
    processed_data = []

    for sentence, entities_json in data:
        entities = entities_json.get('entities', [])
        sentence_entities = []

        processed_sentence = preprocess_text(sentence)

        for entity in entities:
            if ("text" in entity) and ("type" in entity):
                entity_text = entity['text']
                entity_type = entity['type']
                if isinstance(entity_text, list):
                    for single_entity in entity_text:
                        processed_entity_text = preprocess_text(single_entity)

                # Find the starting position of the entity text within the sentence
                        start_index = processed_sentence.find(processed_entity_text)
                        if start_index != -1:
                            end_index = start_index + len(entity_text)
                            sentence_entities.append({
                                'text': processed_entity_text,
                                'type': entity_type,
                                'start': start_index,
                                'end': end_index
                            })
                        else:
                            print(f"Entity '{entity_text}' not found in sentence: '{sentence}'")
                else:
                    processed_entity_text = preprocess_text(entity_text)

                    # Find the starting position of the entity text within the sentence
                    start_index = processed_sentence.find(processed_entity_text)
                    if start_index != -1:
                        end_index = start_index + len(entity_text)
                        sentence_entities.append({
                            'text': processed_entity_text,
                            'type': entity_type,
                            'start': start_index,
                            'end': end_index
                        })
                    else:
                        print(f"Entity '{entity_text}' not found in sentence: '{sentence}'")

        processed_data.append({
            'sentence': processed_sentence,
            'entities': sentence_entities
        })
    return processed_data


def main():
    st.title('NER Visualization Tool')

    # Custom CSS to improve the appearance
    st.markdown("""
            <style>
            .main-container {
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                padding-top: 10px;
            }
            .data-container {
                margin: 5px 0;
                padding: 2px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-radius: 5px;
                background-color: #fafafa;
            }
            .sentence {
                font-size: 12px;
                color: #333;
                line-height: 0.1;
            }
            .entity-list {
                font-size: 14px;
                color: #555;
            }
            </style>
            """, unsafe_allow_html=True)

    items_per_page = 10
    total_items = 1000
    total_pages = (total_items + items_per_page - 1) // items_per_page  # Round up to ensure all items are included

    # Add paginator
    st.sidebar.title("Navigation")
    page_number = st.sidebar.number_input(
        label="Page Number",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1
    )

    # Get start and end indices of the current page
    start_index = (page_number - 1) * items_per_page
    end_index = start_index + items_per_page
    data_lora, data = load_data_two_files(filename1='outputs/output_hg_lora.txt', filename2='outputs/output_hg.txt')
    # data_lora = load_data(xai=False, filename='outputs/output_hg_lora.txt')
    # data = load_data(xai=False, filename='outputs/output_hg.txt')
    processed_data_lora = process_data(data_lora)
    processed_data = process_data(data)
    page_items_lora = processed_data_lora

    nlp = spacy.blank("en")

    # Display each sentence in the current page
    for index in range(start_index + 1, len(page_items_lora)):
        item_lora = page_items_lora[index]
        item = processed_data[index]
        sentence_lora = item_lora['sentence']
        entities_lora = item_lora['entities']
        sentence = item['sentence']
        entities = item['entities']
        entities = [item for item in entities if item['type'] is not None]
        entities_lora = [item for item in entities_lora if item['type'] is not None]

        new_entities_lora = []
        for ent in entities_lora:
            if isinstance(ent['type'], list):
                for label in ent['type']:
                    new_ent = ent.copy()
                    new_ent['type'] = label
                    new_entities_lora.append(new_ent)
            else:
                new_entities_lora.append(ent)

        indices_lora = convert_char_to_token_indices(sentence_lora, new_entities_lora, nlp)

        docs_lora = create_custom_doc(nlp, sentence_lora, indices_lora)
        entity_html_lora = render_entities(docs_lora)

        new_entities = []
        for ent in entities:
            if isinstance(ent['type'], list):
                for label in ent['type']:
                    new_ent = ent.copy()
                    new_ent['type'] = label
                    new_entities.append(new_ent)
            else:
                new_entities.append(ent)

        indices = convert_char_to_token_indices(sentence, new_entities, nlp)

        docs = create_custom_doc(nlp, sentence, indices)
        entity_html = render_entities(docs)

        # Create two columns: one for the highlighted sentence, one for the list of entity types
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f"""
                        <div class='data-container sentence'><strong>Sentence {index} (Akkadian):</strong><br>{entity_html.replace(' - ', '-')}</div>
                        """, unsafe_allow_html=True)
            st.markdown(
                f"""
                        <div class='data-container sentence'><strong>Sentence {index} (Akkadian-Finetuned):</strong><br>{entity_html_lora.replace(' - ', '-')}</div>
                        """, unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='data-container entity-list'><strong>Entities Found:</strong>",
                        unsafe_allow_html=True)
            entity_types = {
                normalize_entity_type(entity['type']): get_color_for_type(normalize_entity_type(entity['type'])) for
                entity in entities}
            for entity_type, color in sorted(entity_types.items()):
                st.markdown(f"<div style='color: {color};'>{entity_type}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
