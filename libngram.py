def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output

def ngrams_text(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(' '.join(input[i:i + n]))
    return output



if __name__ == "__main__":
    for i in range(1,5):
        print ngrams("This is a very big sentence", i)
    for i in range(1,5):
        print ngrams_text("This is a very big sentence", i)

