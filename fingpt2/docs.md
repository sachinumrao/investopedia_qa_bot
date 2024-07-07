## Training GPT2 model using ORPO fine-tuning on finance domain data

1. Accepted response generation

   - [x] Create chunks of investopedia articles
   - [ ] For each chunk, create three QA pairs using Mistral model
   - [ ] Create data with context, question and answer columns
   - [ ] Create train and test set

2. Rejected response geenration

   - [ ] Pass the questions to gpt2-small model and capture model response

3. ORPO gpt2-small model

   - [ ] Preapre fine-tuning script using ORPO trainer in TRL library

4. Model Evaluation

   - [ ] Generate model responsess for test questions
   - [ ] Measure similarity with Mistral model generated responses
