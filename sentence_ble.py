from nltk.translate.bleu_score import sentence_bleu

def str_score(output, expected):
    ### START CODE HERE ###
    output = output.replace('<pad>', '')
    output = output.split('-')
    expected = expected.split('-')
    
    return sentence_bleu([output], expected)
    ### END CODE HERE ###
    
    
def char_score(output, expected):
    ### START CODE HERE ###
    output = output.replace('<pad>', '')
    output = list(output)
    expected = list(expected)
    
    return sentence_bleu([output], expected)
    ### END CODE HERE ###

