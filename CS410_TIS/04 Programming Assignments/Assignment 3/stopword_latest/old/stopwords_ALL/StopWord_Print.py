from stop_words import get_stop_words

stop_words = get_stop_words('en')

f = open("pythonstopwords.txt", "w+")
f.close

for i in range (0, len(stop_words)):
    with open("pythonstopwords.txt", "a") as myfile1:
        myfile1.write(stop_words[i] + "\n")

