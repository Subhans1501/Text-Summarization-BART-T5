# app/ensemble.py

def calculate_ensemble_winner(abstract, generated_titles):
    """
    Generative Ensemble Logic: Post-Hoc Selection
    Evaluates titles from multiple models and mathematically selects the best one 
    based on contextual overlap with the original abstract.
    """
    best_model = None
    best_title = ""
    highest_score = -1
    
    # Clean the abstract into a set of lowercase keywords
    abstract_words = set([word.strip('.,!?"\'') for word in abstract.lower().split()])
    
    # Score each model's output
    for model_name, title in generated_titles.items():
        # Clean the title into a set of lowercase keywords
        title_words = set([word.strip('.,!?"\'') for word in title.lower().split()])
        
        # The Score: How many meaningful words from the title actually appear in the abstract?
        # (We ignore tiny words like 'a', 'the', 'of' by checking length)
        meaningful_title_words = {w for w in title_words if len(w) > 3}
        overlap_score = len(abstract_words.intersection(meaningful_title_words))
        
        if overlap_score > highest_score:
            highest_score = overlap_score
            best_title = title
            best_model = model_name
            
    return best_model, best_title