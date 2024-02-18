from transformers import pipeline
from PyPDF2 import PdfReader


def get_text_from_pdf():
    reader = PdfReader('sample.pdf')
    final_text = ''
    for page in reader.pages:
        final_text += page.extract_text()
    return final_text


def answer_question(context, question):
    pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")
    response = pipe(question=question, context=context)["answer"]
    return response


pdf_text = get_text_from_pdf()
question_to_use = "where can i find more examples of how to use HTML and CSS?"
answer = answer_question(pdf_text, question_to_use)
print(answer)