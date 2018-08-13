import metapy

def tokens_lowercase(doc):
    #Write a token stream that tokenizes with ICUTokenizer, 
    #lowercases, removes words with less than 2 and more than 5  characters
    #performs stemming and creates trigrams (name the final call to ana.analyze as "trigrams")

    tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)     # 1. TOKENIZE
    tok.set_content(doc.content())
    tokens = [token for token in tok]
    print ("The tokens are: ")
    print(tokens)

    tok = metapy.analyzers.LowercaseFilter(tok)                 # 2. LOWERCASE
    tok.set_content(doc.content())
    tokens = [token for token in tok]
    print ("The lower case tokens are: ")
    print(tokens)

    tok = metapy.analyzers.LengthFilter(tok, min=2, max=5)      # 3. REMOVE WORDS < 2 AND > 5 CHAR
    tok.set_content(doc.content())
    tokens = [token for token in tok]
    print ("The tokens after words with < 2 and > 5 char are removed are: ")
    print(tokens)

    tok = metapy.analyzers.Porter2Filter(tok)                   # 4. STEM
    tok.set_content(doc.content())
    tokens = [token for token in tok]
    print ("The tokens after stemming are: ")
    print(tokens)

    tok.set_content(doc.content())
    ana = metapy.analyzers.NGramWordAnalyzer(3, tok)            # 5. CREATE TRIGRAMS
    trigrams = ana.analyze(doc)
    
    #leave the rest of the code as is
    tok.set_content(doc.content())
    tokens, counts = [], []
    for token, count in trigrams.items():
        counts.append(count)
        tokens.append(token)
    return tokens
    
if __name__ == '__main__':
    doc = metapy.index.Document()
    doc.content("I said that I can't believe that it only costs $19.95! I could only find it for more than $30 before.")
    print(doc.content()) #you can access the document string with .content()
    tokens = tokens_lowercase(doc)
    print ("The trigrams are: ")
    print(tokens)
