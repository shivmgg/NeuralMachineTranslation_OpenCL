data_path = './clean_data.txt'

file_row = open('./test_samples.txt', 'w')
file_r = open('./test_actual.txt', 'w')
# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r') as f:
    lines = f.read().split('\n')
for line in lines[6000: 7000]:
    input_text, target_text, _ = line.split('\t')
    file_row.write(input_text + '\n')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = target_text + '\n'
    file_r.write(target_text)

