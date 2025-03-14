"""
NLP Dataset Generator

This script provides utilities to generate synthetic datasets for various NLP tasks:
- Text Classification
- Token Classification
- Table Question Answering
- Question Answering
- Zero-Shot Classification
- Translation
- Summarization
- Feature Extraction
- Text Generation
- Text2Text Generation
- Fill-Mask
- Sentence Similarity
"""

import os
import json
import random
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Sequence
from datasets import Audio, Image

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class NLPDatasetGenerator:
    def __init__(self, output_dir="generated_datasets"):
        """
        Initialize the dataset generator.
        
        Args:
            output_dir (str): Directory to save generated datasets
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_dataset(self, dataset, name):
        """Save the dataset to disk and/or push to Hub."""
        # Save locally
        dataset_path = os.path.join(self.output_dir, name)
        os.makedirs(dataset_path, exist_ok=True)
        dataset.save_to_disk(dataset_path)
        
        # Also save as CSV for easy inspection
        for split in dataset.keys():
            df = dataset[split].to_pandas()
            df.to_csv(os.path.join(dataset_path, f"{split}.csv"), index=False)
            
        print(f"Dataset '{name}' saved to {dataset_path}")
        
        # Optional: Push to Hugging Face Hub
        # dataset.push_to_hub(f"your-username/{name}")
        
    def generate_text_classification(self, num_samples=1000, num_classes=3, class_names=None):
        """
        Generate a dataset for text classification.
        
        Args:
            num_samples (int): Number of samples to generate
            num_classes (int): Number of classes
            class_names (list): Optional list of class names
        """
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]
        else:
            num_classes = len(class_names)
            
        # Template sentences for each class
        templates = {
            0: ["This is a positive review about {topic}.", 
                "I really enjoyed the {topic}, it was amazing!",
                "The {topic} exceeded my expectations."],
            1: ["This is a neutral review about {topic}.", 
                "The {topic} was average, nothing special.",
                "I have mixed feelings about the {topic}."],
            2: ["This is a negative review about {topic}.", 
                "I disliked the {topic} very much.",
                "The {topic} was disappointing and not worth it."]
        }
        
        # Extend templates for more classes if needed
        for i in range(3, num_classes):
            templates[i] = [f"This is a {class_names[i]} type of text.",
                           f"The following is an example of {class_names[i]}.",
                           f"Here's a sample of {class_names[i]}."]
            
        topics = ["movie", "book", "product", "restaurant", "service", 
                 "experience", "concert", "hotel", "app", "game"]
        
        texts = []
        labels = []
        
        for _ in range(num_samples):
            label = random.randint(0, num_classes-1)
            topic = random.choice(topics)
            template = random.choice(templates[min(label, len(templates)-1)])
            text = template.replace("{topic}", topic)
            
            texts.append(text)
            labels.append(label)
            
        # Create dataset
        dataset_dict = {
            "train": Dataset.from_dict({
                "text": texts[:int(num_samples*0.8)], 
                "label": labels[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "text": texts[int(num_samples*0.8):], 
                "label": labels[int(num_samples*0.8):]
            })
        }
        
        # Convert to DatasetDict and set features
        dataset = DatasetDict(dataset_dict)
        dataset = dataset.cast_column("label", ClassLabel(names=class_names))
        
        self.save_dataset(dataset, "text_classification")
        return dataset
        
    def generate_token_classification(self, num_samples=500, entity_types=None):
        """
        Generate a dataset for token classification (NER).
        
        Args:
            num_samples (int): Number of samples to generate
            entity_types (list): Optional list of entity types
        """
        if entity_types is None:
            entity_types = ["PER", "ORG", "LOC", "DATE"]
            
        # Sample entities for each type
        entities = {
            "PER": ["John Smith", "Emma Johnson", "Michael Brown", "Sarah Davis", 
                   "Robert Wilson", "Jennifer Lee", "William Martin", "Lisa Thompson"],
            "ORG": ["Microsoft", "Apple", "Google", "Amazon", "Facebook", 
                   "Tesla", "Netflix", "IBM", "Hugging Face"],
            "LOC": ["New York", "London", "Paris", "Tokyo", "Berlin", 
                   "Sydney", "San Francisco", "Beijing", "Toronto"],
            "DATE": ["January 15, 2023", "May 7, 2022", "December 25, 2021", 
                    "March 10, 2024", "August 3, 2023", "October 21, 2022"]
        }
        
        # Template sentences
        templates = [
            "{PER} works at {ORG} in {LOC} since {DATE}.",
            "{ORG} opened a new office in {LOC} on {DATE}.",
            "{PER} visited {LOC} on {DATE}.",
            "According to {PER}, {ORG} will expand to {LOC} by {DATE}.",
            "The meeting between {PER} and {ORG} representatives took place in {LOC} on {DATE}."
        ]
        
        texts = []
        token_labels = []
        
        for _ in range(num_samples):
            template = random.choice(templates)
            
            # Fill in template with random entities
            filled_template = template
            entities_used = {}
            
            for entity_type in entity_types:
                if entity_type in template and entity_type in entities:
                    entity_value = random.choice(entities[entity_type])
                    filled_template = filled_template.replace("{" + entity_type + "}", entity_value)
                    entities_used[entity_type] = entity_value
            
            # Create tokens and labels
            tokens = []
            labels = []
            
            # Simple tokenization (word-level)
            for word in filled_template.split():
                # Remove punctuation from end of word
                clean_word = word.strip(".,;:!?")
                tokens.append(clean_word)
                
                # Determine label
                found_entity = False
                for entity_type, entity_value in entities_used.items():
                    if clean_word in entity_value.split():
                        if clean_word == entity_value.split()[0]:
                            labels.append(f"B-{entity_type}")
                        else:
                            labels.append(f"I-{entity_type}")
                        found_entity = True
                        break
                
                if not found_entity:
                    labels.append("O")
                    
                # Add punctuation as separate token if it exists
                if word[-1] in ".,;:!?":
                    tokens.append(word[-1])
                    labels.append("O")
            
            texts.append(filled_template)
            token_labels.append({"tokens": tokens, "ner_tags": labels})
        
        # Create dataset
        dataset_dict = {
            "train": Dataset.from_dict({
                "text": texts[:int(num_samples*0.8)],
                "tokens": [t["tokens"] for t in token_labels[:int(num_samples*0.8)]],
                "ner_tags": [t["ner_tags"] for t in token_labels[:int(num_samples*0.8)]]
            }),
            "test": Dataset.from_dict({
                "text": texts[int(num_samples*0.8):],
                "tokens": [t["tokens"] for t in token_labels[int(num_samples*0.8):]],
                "ner_tags": [t["ner_tags"] for t in token_labels[int(num_samples*0.8):]]
            })
        }
        
        # Convert to DatasetDict
        dataset = DatasetDict(dataset_dict)
        
        self.save_dataset(dataset, "token_classification")
        return dataset
        
    def generate_question_answering(self, num_samples=500):
        """Generate a dataset for extractive question answering."""
        
        # Templates for contexts
        context_templates = [
            "The capital of France is Paris. It is known for landmarks such as the Eiffel Tower and the Louvre Museum. France is located in Western Europe.",
            "Albert Einstein was born on March 14, 1879, in Ulm, Germany. He developed the theory of relativity and won the Nobel Prize in Physics in 1921.",
            "The Pacific Ocean is the largest and deepest ocean on Earth. It covers more than 60 million square miles and contains more than half of the free water on Earth.",
            "The human body has 206 bones. The smallest bone is the stirrup bone located in the middle ear. The longest bone is the femur, or thigh bone.",
            "Photosynthesis is the process by which green plants use sunlight to synthesize foods with carbon dioxide and water. It produces oxygen as a byproduct."
        ]
        
        # Extend with more contexts for variety
        more_contexts = [
            "The Great Wall of China is one of the Seven Wonders of the World. Construction began in the 7th century BC and continued for over 2,000 years. It stretches approximately 13,171 miles.",
            "Water boils at 100 degrees Celsius at standard atmospheric pressure. When water reaches its boiling point, it converts from liquid to gas state, creating bubbles and steam.",
            "The first computer programmer was Ada Lovelace, who wrote an algorithm for Charles Babbage's Analytical Engine in the 1840s. The programming language Ada was named after her.",
            "Mount Everest is the highest mountain in the world, with a peak at 29,032 feet above sea level. It is located in the Mahalangur Himal sub-range of the Himalayas.",
            "The human brain contains approximately 86 billion neurons. It consumes about 20% of the body's oxygen and calories despite only accounting for 2% of total body weight."
        ]
        
        context_templates.extend(more_contexts)
        
        # Generate QA pairs
        contexts = []
        questions = []
        answers = []
        
        for _ in range(num_samples):
            # Choose a random context
            context = random.choice(context_templates)
            contexts.append(context)
            
            # Generate a question and answer from context
            sentences = context.split(". ")
            target_sentence = random.choice(sentences)
            
            # Extract key information for the question
            words = target_sentence.split()
            if len(words) < 3:
                continue
                
            # Determine question type
            question_type = random.choice(["what", "where", "who", "when"])
            
            if question_type == "what":
                # Extract a noun from the sentence as the answer
                nouns = [w for w in words if len(w) > 3 and w[0].isupper() or w in ["ocean", "bone", "brain", "water", "theory"]]
                if nouns:
                    answer_text = random.choice(nouns)
                    question_text = target_sentence.replace(answer_text, "what")
                    question_text = f"What {question_text.split('what', 1)[1].strip()}?"
                else:
                    answer_text = words[-2]
                    question_text = f"What is mentioned in relation to {' '.join(words[:-2])}?"
            elif question_type == "where":
                locations = [w for w in words if w in ["France", "Europe", "Germany", "Earth", "China", "Himalayas", "Ulm"]]
                if locations:
                    answer_text = random.choice(locations)
                    question_text = f"Where is {words[0]} located?" if words[0] != answer_text else "Where was this event taking place?"
                else:
                    answer_text = words[-1]
                    question_text = f"Where did the event involving {words[0]} happen?"
            elif question_type == "who":
                people = [w for w in words if w in ["Einstein", "Lovelace", "Ada"]]
                if people:
                    answer_text = random.choice(people)
                    question_text = f"Who is mentioned in relation to {random.choice(words)}?"
                else:
                    answer_text = words[0] if words[0][0].isupper() else words[-1]
                    question_text = f"Who is the subject of this text?"
            else: # when
                dates = [w for w in words if w in ["1879", "1921", "1840s", "March", "century"]]
                if dates:
                    answer_text = random.choice(dates)
                    question_text = f"When did the event involving {words[0]} happen?"
                else:
                    answer_text = words[-1]
                    question_text = "When did this event occur?"
            
            # Find the answer's character position in context
            answer_start = context.find(answer_text)
            if answer_start == -1:
                # Fallback if exact match not found
                answer_text = words[-1]
                answer_start = context.find(answer_text)
            
            questions.append(question_text)
            answers.append({"text": answer_text, "answer_start": answer_start})
            
        # Create dataset
        dataset_dict = {
            "train": Dataset.from_dict({
                "context": contexts[:int(num_samples*0.8)],
                "question": questions[:int(num_samples*0.8)],
                "answers": answers[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "context": contexts[int(num_samples*0.8):],
                "question": questions[int(num_samples*0.8):],
                "answers": answers[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "question_answering")
        return dataset
        
    def generate_summarization(self, num_samples=500):
        """Generate a dataset for text summarization."""
        
        # Templates for articles and their summaries
        article_templates = [
            {
                "article": """The city council met on Tuesday to discuss the new public transportation plan. 
                The proposed plan includes extending the subway line to the suburbs and increasing bus frequency during peak hours.
                Several council members expressed concerns about the cost, estimated at $50 million over five years.
                Citizens who attended the meeting were largely supportive, citing the need for better commuting options.
                The council decided to form a committee to further study the proposal before voting next month.""",
                
                "summary": "City council discussed a $50 million public transportation expansion plan and formed a committee to study it further before next month's vote."
            },
            {
                "article": """Researchers at the university have developed a new vaccine that shows promise in treating multiple strains of influenza.
                The vaccine, which has been in development for over five years, uses a novel approach to target proteins common to all flu strains.
                In clinical trials involving 1,000 participants, the vaccine demonstrated 87% effectiveness against various flu strains.
                Side effects were minimal and comparable to existing vaccines currently on the market.
                The research team hopes to receive regulatory approval within two years if additional trials continue to show positive results.""",
                
                "summary": "University researchers developed a promising multi-strain flu vaccine with 87% effectiveness in trials that could receive approval within two years."
            }
        ]
        
        # Add more templates
        more_templates = [
            {
                "article": """A major technological breakthrough was announced yesterday by Tech Innovations Inc., a leading technology company based in Silicon Valley.
                Their new battery technology can reportedly store twice as much energy as current lithium-ion batteries while charging three times faster.
                The company claims the batteries can withstand over 1,000 charge cycles without significant degradation, far exceeding industry standards.
                Environmental experts have praised the development, noting that the new batteries use fewer rare earth minerals and are more recyclable.
                Production is expected to begin next year, with the first consumer products featuring the technology potentially available by 2026.""",
                
                "summary": "Tech Innovations Inc. announced a new battery technology with double the energy storage and triple the charging speed of current batteries, with production starting next year."
            },
            {
                "article": """Global temperatures have risen by an average of 1.2 degrees Celsius since pre-industrial times, according to a new climate report released this week.
                The report, compiled by scientists from 50 countries, warns that if current trends continue, we could see a 3-degree increase by 2100.
                Rising sea levels, more frequent extreme weather events, and widespread ecosystem disruption are among the predicted consequences.
                The report calls for immediate action to reduce carbon emissions by at least 50% by 2030 to avoid the worst effects.
                Over 190 nations have been urged to strengthen their climate commitments ahead of next year's global climate conference.""",
                
                "summary": "A new climate report shows global temperatures have risen 1.2°C and could increase to 3°C by 2100, calling for 50% carbon emission reductions by 2030."
            },
            {
                "article": """A landmark trade agreement was reached yesterday between the neighboring countries of Eastland and Westland after three years of negotiations.
                The deal eliminates tariffs on agricultural products and reduces barriers for technology and automotive sectors.
                Economists project the agreement will increase bilateral trade by approximately $30 billion annually and create an estimated 250,000 jobs across both nations.
                Opposition parties in both countries have criticized the deal, claiming it doesn't include strong enough environmental and labor protections.
                The agreement requires ratification by both countries' legislatures and is expected to take effect in January if approved.""",
                
                "summary": "Eastland and Westland reached a trade agreement eliminating agricultural tariffs and reducing barriers for tech and automotive sectors, potentially increasing trade by $30 billion annually."
            }
        ]
        
        article_templates.extend(more_templates)
        
        # Generate article-summary pairs
        articles = []
        summaries = []
        
        for _ in range(num_samples):
            template = random.choice(article_templates)
            
            # Add some variation to the articles
            article = template["article"]
            summary = template["summary"]
            
            # Occasionally add/remove sentences or modify details
            if random.random() > 0.7:
                sentences = article.split(". ")
                if len(sentences) > 3:
                    if random.random() > 0.5:
                        # Remove a random sentence
                        del sentences[random.randint(1, len(sentences)-2)]
                    else:
                        # Add a generic sentence
                        generic_sentences = [
                            "Experts continue to debate the long-term implications",
                            "This development follows years of careful planning and research",
                            "Public reaction has been mixed but generally positive",
                            "Similar initiatives have been attempted in the past with varying degrees of success"
                        ]
                        sentences.insert(random.randint(1, len(sentences)-1), random.choice(generic_sentences))
                
                article = ". ".join(sentences)
            
            articles.append(article)
            summaries.append(summary)
            
        # Create dataset
        dataset_dict = {
            "train": Dataset.from_dict({
                "article": articles[:int(num_samples*0.8)],
                "summary": summaries[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "article": articles[int(num_samples*0.8):],
                "summary": summaries[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "summarization")
        return dataset
    
    def generate_translation(self, num_samples=500, language_pairs=None):
        """Generate a dataset for machine translation."""
        if language_pairs is None:
            language_pairs = [("en", "fr"), ("en", "es")]
            
        # Templates for translations
        translation_templates = {
            ("en", "fr"): [
                {"en": "Hello, how are you?", "fr": "Bonjour, comment allez-vous?"},
                {"en": "I would like to order a coffee.", "fr": "Je voudrais commander un café."},
                {"en": "Where is the nearest hospital?", "fr": "Où est l'hôpital le plus proche?"},
                {"en": "What time does the train leave?", "fr": "À quelle heure part le train?"},
                {"en": "Thank you for your help.", "fr": "Merci pour votre aide."},
                {"en": "My name is John Smith.", "fr": "Je m'appelle John Smith."},
                {"en": "I don't understand French very well.", "fr": "Je ne comprends pas très bien le français."},
                {"en": "Can you speak more slowly please?", "fr": "Pouvez-vous parler plus lentement s'il vous plaît?"},
                {"en": "How much does this cost?", "fr": "Combien ça coûte?"},
                {"en": "I need to buy a ticket.", "fr": "J'ai besoin d'acheter un billet."}
            ],
            ("en", "es"): [
                {"en": "Hello, how are you?", "es": "Hola, ¿cómo estás?"},
                {"en": "I would like to order a coffee.", "es": "Me gustaría pedir un café."},
                {"en": "Where is the nearest hospital?", "es": "¿Dónde está el hospital más cercano?"},
                {"en": "What time does the train leave?", "es": "¿A qué hora sale el tren?"},
                {"en": "Thank you for your help.", "es": "Gracias por tu ayuda."},
                {"en": "My name is John Smith.", "es": "Me llamo John Smith."},
                {"en": "I don't understand Spanish very well.", "es": "No entiendo muy bien el español."},
                {"en": "Can you speak more slowly please?", "es": "¿Puedes hablar más despacio por favor?"},
                {"en": "How much does this cost?", "es": "¿Cuánto cuesta esto?"},
                {"en": "I need to buy a ticket.", "es": "Necesito comprar un boleto."}
            ]
        }
        
        all_translations = []
        
        for _ in range(num_samples):
            language_pair = random.choice(language_pairs)
            if language_pair not in translation_templates:
                continue
                
            template = random.choice(translation_templates[language_pair])
            source_lang, target_lang = language_pair
            
            # Add some variation to translations
            translation = {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "source": template[source_lang],
                "target": template[target_lang]
            }
            
            all_translations.append(translation)
            
        # Create datasets for each language pair
        for source_lang, target_lang in language_pairs:
            pair_translations = [t for t in all_translations if t["source_lang"] == source_lang and t["target_lang"] == target_lang]
            if not pair_translations:
                continue
                
            dataset_dict = {
                "train": Dataset.from_dict({
                    "translation": [
                        {source_lang: t["source"], target_lang: t["target"]} 
                        for t in pair_translations[:int(len(pair_translations)*0.8)]
                    ]
                }),
                "test": Dataset.from_dict({
                    "translation": [
                        {source_lang: t["source"], target_lang: t["target"]} 
                        for t in pair_translations[int(len(pair_translations)*0.8):]
                    ]
                })
            }
            
            dataset = DatasetDict(dataset_dict)
            self.save_dataset(dataset, f"translation_{source_lang}_to_{target_lang}")
            
        return dataset

    def generate_sentence_similarity(self, num_samples=500):
        """Generate a dataset for sentence similarity."""
        
        # Templates for sentence pairs with similarity scores
        sentence_pairs = [
            {
                "sentence1": "The cat is playing with the yarn.",
                "sentence2": "The feline is toying with the string.",
                "similarity": 0.9
            },
            {
                "sentence1": "The cat is playing with the yarn.",
                "sentence2": "The dog is sleeping on the couch.",
                "similarity": 0.1
            },
            {
                "sentence1": "I enjoy reading books in my free time.",
                "sentence2": "In my spare time, I like to read books.",
                "similarity": 0.95
            },
            {
                "sentence1": "I enjoy reading books in my free time.",
                "sentence2": "I prefer watching movies instead of reading.",
                "similarity": 0.4
            },
            {
                "sentence1": "The stock market crashed yesterday.",
                "sentence2": "Share prices plummeted in yesterday's trading.",
                "similarity": 0.85
            },
            {
                "sentence1": "The stock market crashed yesterday.",
                "sentence2": "The company announced a new product line.",
                "similarity": 0.15
            },
        ]
        
        # Additional pairs
        more_pairs = [
            {
                "sentence1": "She made a delicious chocolate cake.",
                "sentence2": "A chocolate cake was baked by her.",
                "similarity": 0.8
            },
            {
                "sentence1": "She made a delicious chocolate cake.",
                "sentence2": "The weather is nice today.",
                "similarity": 0.0
            },
            {
                "sentence1": "The train will arrive at 3 PM.",
                "sentence2": "The train arrives at three o'clock in the afternoon.",
                "similarity": 0.9
            },
            {
                "sentence1": "The train will arrive at 3 PM.",
                "sentence2": "I missed my flight yesterday.",
                "similarity": 0.2
            },
        ]
        
        sentence_pairs.extend(more_pairs)
        
        # Generate sentence pairs with similarity scores
        sentence1_list = []
        sentence2_list = []
        similarity_scores = []
        
        for _ in range(num_samples):
            pair = random.choice(sentence_pairs)
            
            # Add some variation
            if random.random() > 0.5:
                # Use pair as is
                sentence1 = pair["sentence1"]
                sentence2 = pair["sentence2"]
                similarity = pair["similarity"]
            else:
                # Mix and match sentences from different pairs
                pair2 = random.choice(sentence_pairs)
                if random.random() > 0.5:
                    sentence1 = pair["sentence1"]
                    sentence2 = pair2["sentence2"]
                    # Assign a random lower similarity score for mixed pairs
                    similarity = random.uniform(0.0, 0.5)
                else:
                    # Use similar sentences
                    similar_pairs = [p for p in sentence_pairs if abs(p["similarity"] - pair["similarity"]) < 0.2]
                    if similar_pairs:
                        pair2 = random.choice(similar_pairs)
                        sentence1 = pair["sentence1"]
                        sentence2 = pair2["sentence2"]
                        similarity = (pair["similarity"] + pair2["similarity"]) / 2
                    else:
                        # Fallback to original pair
                        sentence1 = pair["sentence1"]
                        sentence2 = pair["sentence2"]
                        similarity = pair["similarity"]
            
            sentence1_list.append(sentence1)
            sentence2_list.append(sentence2)
            similarity_scores.append(similarity)
            
        # Create dataset
        dataset_dict = {
            "train": Dataset.from_dict({
                "sentence1": sentence1_list[:int(num_samples*0.8)],
                "sentence2": sentence2_list[:int(num_samples*0.8)],
                "similarity": similarity_scores[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "sentence1": sentence1_list[int(num_samples*0.8):],
                "sentence2": sentence2_list[int(num_samples*0.8):],
                "similarity": similarity_scores[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "sentence_similarity")
        return dataset
    
    def generate_fill_mask(self, num_samples=500):
        """Generate a dataset for masked language modeling (fill-mask task)."""
        
        # Templates with masks
        mask_token = "[MASK]"
        templates = [
            f"The capital of France is {mask_token}.",
            f"She enjoys reading {mask_token} in her free time.",
            f"The {mask_token} is the largest organ in the human body.",
            f"Water boils at {mask_token} degrees Celsius.",
            f"{mask_token} is the current president of the United States.",
            f"The Eiffel Tower is located in {mask_token}.",
            f"Humans have {mask_token} fingers on each hand.",
            f"The Earth revolves around the {mask_token}.",
            f"The most abundant gas in Earth's atmosphere is {mask_token}.",
            f"Shakespeare wrote the play {mask_token}."
        ]
        
        # Answers for each template
        answers = [
            ["Paris", "Lyon", "Marseille"],
            ["books", "novels", "magazines", "newspapers"],
            ["skin", "liver", "brain"],
            ["100", "ninety-nine", "212"],
            ["Biden", "Joe Biden"],
            ["Paris", "France", "Europe"],
            ["five", "5", "four"],
            ["sun", "star", "Sun"],
            ["nitrogen", "oxygen", "carbon dioxide"],
            ["Hamlet", "Macbeth", "Romeo and Juliet"]
        ]
        
        # Generate masked sentences and their possible completions
        masked_texts = []
        original_texts = []
        mask_positions = []
        
        for _ in range(num_samples):
            template_idx = random.randint(0, len(templates) - 1)
            template = templates[template_idx]
            answer = random.choice(answers[template_idx])
            
            # Find mask position
            mask_pos = template.find(mask_token)
            
                        # Replace mask with answer to get original text
            original_text = template.replace(mask_token, answer)
            
            masked_texts.append(template)
            original_texts.append(original_text)
            mask_positions.append(mask_pos)
            
        # Create dataset
        dataset_dict = {
            "train": Dataset.from_dict({
                "masked_text": masked_texts[:int(num_samples*0.8)],
                "original_text": original_texts[:int(num_samples*0.8)],
                "mask_position": mask_positions[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "masked_text": masked_texts[int(num_samples*0.8):],
                "original_text": original_texts[int(num_samples*0.8):],
                "mask_position": mask_positions[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "fill_mask")
        return dataset
    
    def generate_zero_shot_classification(self, num_samples=500):
        """Generate a dataset for zero-shot classification."""
        
        # Define candidate labels for different topics
        candidate_labels = {
            "sentiment": ["positive", "negative", "neutral"],
            "topic": ["politics", "sports", "technology", "entertainment", "business"],
            "emotion": ["joy", "sadness", "anger", "fear", "surprise"],
            "intent": ["inquiry", "complaint", "suggestion", "praise", "request"]
        }
        
        # Templates for texts covering different topics
        text_templates = {
            "sentiment": [
                "I absolutely loved the movie, it was fantastic!",
                "The service at the restaurant was terrible and the food was cold.",
                "The product works as expected, nothing special to report.",
                "This is the best book I've read all year, highly recommended.",
                "I was disappointed with the quality of the item when it arrived."
            ],
            "topic": [
                "The latest election results show a significant shift in voter preferences compared to the last cycle.",
                "The team scored in the final minutes to win the championship after a tough season.",
                "The new smartphone features a foldable display and improved battery life.",
                "The movie won several awards at the film festival and critics praised the performances.",
                "Stock markets declined sharply following the release of the latest economic data."
            ],
            "emotion": [
                "I just got promoted at work! This is the best day ever!",
                "I miss my family so much, it's been months since I've seen them.",
                "How could they make such a decision without consulting me first?!",
                "I'm worried about the upcoming exam, I don't feel prepared at all.",
                "Wow! I never expected to receive such an amazing gift!"
            ],
            "intent": [
                "Could you please tell me how to reset my password?",
                "Your customer service is absolutely unacceptable and I want a refund immediately.",
                "It might be helpful if you added a dark mode to the app.",
                "I just wanted to say that your team did an excellent job with the project.",
                "I need assistance with setting up my new account as soon as possible."
            ]
        }
        
        # Generate zero-shot classification examples
        texts = []
        label_candidates = []
        categories = []
        
        for _ in range(num_samples):
            # Select a random category (sentiment, topic, etc.)
            category = random.choice(list(candidate_labels.keys()))
            
            # Select a text template for that category
            text = random.choice(text_templates[category])
            
            # Get candidate labels for that category
            candidates = candidate_labels[category]
            
            texts.append(text)
            label_candidates.append(candidates)
            categories.append(category)
            
        # Create dataset
        dataset_dict = {
            "train": Dataset.from_dict({
                "text": texts[:int(num_samples*0.8)],
                "candidate_labels": label_candidates[:int(num_samples*0.8)],
                "category": categories[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "text": texts[int(num_samples*0.8):],
                "candidate_labels": label_candidates[int(num_samples*0.8):],
                "category": categories[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "zero_shot_classification")
        return dataset
        
    def generate_text_generation(self, num_samples=300):
        """Generate a dataset for text generation (prompts for generation)."""
        
        # Templates for text generation prompts
        prompt_templates = [
            "Once upon a time, there was a",
            "The best way to solve this problem is to",
            "In the future, artificial intelligence will",
            "If I could travel anywhere in the world, I would go to",
            "The secret to happiness is",
            "The most important invention of the last century was",
            "When I look back on my life, I remember",
            "The three things I value most in life are",
            "The biggest challenge facing humanity today is",
            "My favorite childhood memory is",
            "If I could have any superpower, it would be",
            "The meaning of life is",
            "In my opinion, the best book ever written is",
            "Ten years from now, I hope to",
            "The most beautiful place I've ever visited was"
        ]
        
        # Some example completions (for reference, not included in dataset)
        completions = {
            "Once upon a time, there was a": " young girl who lived at the edge of a forest. She loved to explore and discover new things in nature.",
            "The best way to solve this problem is to": " break it down into smaller, manageable steps and tackle each one methodically.",
            "In the future, artificial intelligence will": " help us solve complex problems in medicine, climate science, and space exploration."
        }
        
        # Generate prompts
        prompts = []
        max_lengths = []
        
        for _ in range(num_samples):
            prompt = random.choice(prompt_templates)
            
            # Add some variation to prompts
            if random.random() > 0.7:
                words = prompt.split()
                if len(words) > 4:
                    # Modify the prompt slightly
                    if random.random() > 0.5:
                        # Change an adjective or noun
                        replacements = {
                            "best": "quickest", "secret": "key", "biggest": "greatest",
                            "problem": "challenge", "important": "significant", "favorite": "cherished"
                        }
                        for old, new in replacements.items():
                            if old in words:
                                words[words.index(old)] = new
                                break
                
                prompt = " ".join(words)
            
            # Random expected generation length
            max_length = random.randint(50, 200)
            
            prompts.append(prompt)
            max_lengths.append(max_length)
            
        # Create dataset
        dataset_dict = {
            "train": Dataset.from_dict({
                "prompt": prompts[:int(num_samples*0.8)],
                "max_length": max_lengths[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "prompt": prompts[int(num_samples*0.8):],
                "max_length": max_lengths[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "text_generation")
        return dataset
        
    def generate_text2text_generation(self, num_samples=500):
        """Generate a dataset for text-to-text generation tasks like style transfer or paraphrasing."""
        
        # Templates for source and target pairs
        templates = [
            # Formal to informal
            {
                "source": "I would greatly appreciate it if you could provide me with that information at your earliest convenience.",
                "target": "Could you give me that info when you get a chance?",
                "task": "formal_to_informal"
            },
            {
                "source": "We regret to inform you that your application has been declined.",
                "target": "Sorry, we can't accept your application.",
                "task": "formal_to_informal"
            },
            
            # Informal to formal
            {
                "source": "Hey, can you get this done by tomorrow?",
                "target": "I kindly request that you complete this task by tomorrow.",
                "task": "informal_to_formal"
            },
            {
                "source": "This movie is awesome!",
                "target": "This film is exceptionally enjoyable and of high quality.",
                "task": "informal_to_formal"
            },
            
            # Active to passive
            {
                "source": "The chef prepared a delicious meal.",
                "target": "A delicious meal was prepared by the chef.",
                "task": "active_to_passive"
            },
            {
                "source": "The company launched a new product last month.",
                "target": "A new product was launched by the company last month.",
                "task": "active_to_passive"
            },
            
            # Passive to active
            {
                "source": "The book was written by a famous author.",
                "target": "A famous author wrote the book.",
                "task": "passive_to_active"
            },
            {
                "source": "The road was damaged by the storm.",
                "target": "The storm damaged the road.",
                "task": "passive_to_active"
            },
            
            # Simplification
            {
                "source": "The physician utilized the stethoscope to auscultate the patient's cardiovascular system.",
                "target": "The doctor used the stethoscope to listen to the patient's heart.",
                "task": "simplification"
            },
            {
                "source": "Numerous individuals congregated to observe the celestial phenomenon.",
                "target": "Many people gathered to watch the event in the sky.",
                "task": "simplification"
            }
        ]
        
        # Add more examples
        more_examples = [
            # Formal to informal
            {
                "source": "We cordially invite you to attend our annual celebration.",
                "target": "We'd love for you to come to our yearly party.",
                "task": "formal_to_informal"
            },
            
            # Informal to formal
            {
                "source": "This place is totally cool!",
                "target": "This establishment is quite impressive.",
                "task": "informal_to_formal"
            },
            
            # Active to passive
            {
                "source": "Students submit assignments online.",
                "target": "Assignments are submitted online by students.",
                "task": "active_to_passive"
            },
            
            # Passive to active
            {
                "source": "The window was broken by the baseball.",
                "target": "The baseball broke the window.",
                "task": "passive_to_active"
            },
            
            # Simplification
            {
                "source": "The implementation of the innovative agricultural techniques resulted in a substantial augmentation in crop yield.",
                "target": "Using new farming methods led to much bigger harvests.",
                "task": "simplification"
            }
        ]
        
        templates.extend(more_examples)
        
        # Generate text2text pairs
        sources = []
        targets = []
        tasks = []
        
        for _ in range(num_samples):
            template = random.choice(templates)
            
            sources.append(template["source"])
            targets.append(template["target"])
            tasks.append(template["task"])
            
        # Create dataset
        dataset_dict = {
            "train": Dataset.from_dict({
                "source": sources[:int(num_samples*0.8)],
                "target": targets[:int(num_samples*0.8)],
                "task": tasks[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "source": sources[int(num_samples*0.8):],
                "target": targets[int(num_samples*0.8):],
                "task": tasks[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "text2text_generation")
        return dataset
    
    def generate_table_qa(self, num_samples=300):
        """Generate a dataset for table question answering."""
        
        # Create some template tables
        tables = [
            # Sales table
            {
                "header": ["Product", "Q1", "Q2", "Q3", "Q4", "Total"],
                "rows": [
                    ["Laptops", "150", "200", "180", "220", "750"],
                    ["Phones", "320", "280", "350", "400", "1350"],
                    ["Tablets", "90", "120", "85", "100", "395"],
                    ["Accessories", "430", "400", "450", "500", "1780"]
                ],
                "questions": [
                    "What were the total sales of Phones?",
                    "Which product had the highest sales in Q2?",
                    "How many Laptops were sold in Q3?",
                    "What was the total number of Tablets sold across all quarters?",
                    "What is the difference between the highest and lowest quarterly sales for Accessories?"
                ],
                "answers": [
                    "1350",
                    "Phones",
                    "180",
                    "395",
                    "100"
                ]
            },
            
            # Employee table
            {
                "header": ["Name", "Department", "Position", "Salary", "Start Date"],
                "rows": [
                    ["John Smith", "Engineering", "Senior Developer", "$120,000", "2018-05-15"],
                    ["Emma Johnson", "Marketing", "Marketing Manager", "$95,000", "2019-03-10"],
                    ["Michael Brown", "Finance", "Financial Analyst", "$85,000", "2020-01-22"],
                    ["Sarah Davis", "HR", "HR Specialist", "$78,000", "2017-11-08"],
                    ["Robert Wilson", "Engineering", "Junior Developer", "$75,000", "2021-07-30"]
                ],
                "questions": [
                    "Who is the Marketing Manager?",
                    "What is John Smith's position?",
                    "When did Sarah Davis join the company?",
                    "What is the salary of the Junior Developer?",
                    "Which department does Michael Brown work in?"
                ],
                "answers": [
                    "Emma Johnson",
                    "Senior Developer",
                    "2017-11-08",
                    "$75,000",
                    "Finance"
                ]
            },
            
            # Country data table
            {
                "header": ["Country", "Population (millions)", "GDP ($ billions)", "Capital", "Continent"],
                "rows": [
                    ["USA", "331", "21,400", "Washington D.C.", "North America"],
                    ["China", "1,400", "14,300", "Beijing", "Asia"],
                    ["India", "1,366", "2,870", "New Delhi", "Asia"],
                    ["Germany", "83", "3,860", "Berlin", "Europe"],
                    ["Brazil", "212", "1,870", "Brasília", "South America"]
                ],
                "questions": [
                    "What is the capital of India?",
                    "Which country has the largest population?",
                    "What is the GDP of Germany?",
                    "Which continent has the most countries listed in the table?",
                    "What is the population of Brazil?"
                ],
                "answers": [
                    "New Delhi",
                    "China",
                    "3,860",
                    "Asia",
                    "212"
                ]
            }
        ]
        
        # Generate table QA examples
        table_dicts = []
        questions = []
        answers = []
        
        for _ in range(num_samples):
            table = random.choice(tables)
            
            # Randomly select a question-answer pair
            qa_idx = random.randint(0, len(table["questions"]) - 1)
            question = table["questions"][qa_idx]
            answer = table["answers"][qa_idx]
            
            # Create table dict
            table_dict = {
                "header": table["header"],
                "rows": table["rows"]
            }
            
            table_dicts.append(table_dict)
            questions.append(question)
            answers.append(answer)
            
        # Create dataset
        dataset_dict = {
            "train": Dataset.from_dict({
                "table": table_dicts[:int(num_samples*0.8)],
                "question": questions[:int(num_samples*0.8)],
                "answer": answers[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "table": table_dicts[int(num_samples*0.8):],
                "question": questions[int(num_samples*0.8):],
                "answer": answers[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "table_qa")
        return dataset
        
    def generate_feature_extraction(self, num_samples=300):
        """Generate a dataset for feature extraction (text embeddings)."""
        
        # Define text categories and examples
        categories = {
            "science": [
                "The theory of relativity explains the relationship between space and time.",
                "Quantum mechanics deals with the behavior of matter at the atomic and subatomic scale.",
                "Photosynthesis is the process by which plants convert light energy into chemical energy.",
                "DNA contains the genetic instructions used in the development and functioning of all living organisms.",
                "The periodic table organizes chemical elements by their atomic number and chemical properties."
            ],
            "arts": [
                "The Mona Lisa is a famous portrait painting created by Leonardo da Vinci.",
                "Impressionism focused on capturing light and color rather than precise details.",
                "Classical music evolved during the Classical period from 1730 to 1820.",
                "Literature encompasses written works, particularly those considered to have artistic merit.",
                "Ballet is a type of performance dance that originated during the Italian Renaissance."
            ],
            "technology": [
                "Machine learning is a method of data analysis that automates analytical model building.",
                "Cloud computing provides on-demand availability of computer system resources.",
                "Blockchain is a system of recording information in a way that makes it difficult to change or hack.",
                "Virtual reality creates a simulated environment that can be similar to or completely different from the real world.",
                "The Internet of Things refers to physical objects embedded with sensors, software, and other technologies."
            ],
            "history": [
                "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity.",
                "The Industrial Revolution was the transition to new manufacturing processes in Europe and the United States.",
                "World War II was a global war that lasted from 1939 to 1945, involving many of the world's nations.",
                "The Cold War was a period of geopolitical tension between the Soviet Union and the United States.",
                "Ancient Egypt was a civilization of ancient North Africa, concentrated along the lower Nile River."
            ]
        }
        
        # Generate text examples and their categories
        texts = []
        text_categories = []
        
        for _ in range(num_samples):
            # Select a random category
            category = random.choice(list(categories.keys()))
            
            # Select a random example from that category
            text = random.choice(categories[category])
            
            texts.append(text)
            text_categories.append(category)
            
        # Create dataset
        dataset_dict = {
            "train": Dataset.from_dict({
                "text": texts[:int(num_samples*0.8)],
                "category": text_categories[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "text": texts[int(num_samples*0.8):],
                "category": text_categories[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "feature_extraction")
        return dataset

    def generate_all_datasets(self, samples_per_task=500):
        """Generate all supported dataset types."""
        print(f"Generating all NLP datasets with {samples_per_task} samples per task...")
        
        self.generate_text_classification(num_samples=samples_per_task)
        self.generate_token_classification(num_samples=samples_per_task)
        self.generate_question_answering(num_samples=samples_per_task)
        self.generate_summarization(num_samples=samples_per_task)
        self.generate_translation(num_samples=samples_per_task)
        self.generate_sentence_similarity(num_samples=samples_per_task)
        self.generate_fill_mask(num_samples=samples_per_task)
        self.generate_zero_shot_classification(num_samples=samples_per_task)
        self.generate_text_generation(num_samples=samples_per_task)
        self.generate_text2text_generation(num_samples=samples_per_task)
        self.generate_table_qa(num_samples=samples_per_task)
        self.generate_feature_extraction(num_samples=samples_per_task)
        
        print("All datasets generated successfully!")