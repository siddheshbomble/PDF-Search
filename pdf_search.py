import PyPDF2
from transformers import pipeline

# Load the pre-trained question-answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

pdf_path = r"/content/MyResume.pdf"  # Path


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file) 
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
    return text


def answer_question(text, question):
    result = qa_pipeline(question=question, context=text)
    return result.get('answer', 'No answer found.')


def main():
    pdf_path = r"/content/MyResume.pdf"  # Path
    text = extract_text_from_pdf(pdf_path)

    if not text:
        print("No text found in the PDF.")
        return

    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = answer_question(text, question)
        print("Answer:", answer)


if __name__ == "__main__":
    main()
