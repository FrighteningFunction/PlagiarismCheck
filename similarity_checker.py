import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_files(directory):
    files_content = []
    file_names = []
    for filename in os.listdir(directory):
        if filename.endswith(".cpp"):  # Check for .cpp files
            path = os.path.join(directory, filename)
            try:
                # Try reading with UTF-8
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.read()
            except UnicodeDecodeError:
                # Fallback to Latin-1 if UTF-8 fails
                with open(path, 'r', encoding='iso-8859-1') as file:
                    content = file.read()
            files_content.append(content)
            file_names.append(filename)
    return file_names, files_content


def calculate_similarity(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

def main():
    directory = 'code_submissions'  # Ensure this directory is correctly specified
    file_names, files_content = read_files(directory)
    if files_content:
        similarity_matrix = calculate_similarity(files_content)
        print("Similarity Matrix:")
        for i in range(len(file_names)):
            for j in range(i + 1, len(file_names)):
                print(f"{file_names[i]} vs {file_names[j]}: {similarity_matrix[i][j]}")
    else:
        print("No .cpp files found in the directory.")  # Inform if no .cpp files are found

if __name__ == "__main__":
    main()
