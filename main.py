from transformers import pipeline
import streamlit as st


def get_text():
    return st.text_area(
        label="Введите предложение на английском или французском языках(желаемое для подбора слово замените на <mask>):",
        value="My name is Boris and i'm from <mask>."
    )


def main():
    st.title("")
    text = get_text()
    camembert_fill_mask = pipeline("fill-mask", model="camembert-base", tokenizer="camembert-base")

    if text != "":
        results = camembert_fill_mask(text)
        st.caption("Варианты слов и их вероятностное распределение:")
        for el in results:
            st.write(el["token_str"] + " - " + str(round(el["score"] * 100)) + "%")


if __name__ == "__main__":
    main()
