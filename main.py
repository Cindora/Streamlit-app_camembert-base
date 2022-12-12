from transformers import pipeline
import streamlit as st


def get_text():
    return st.text_area(
        label="Введите предложение на английском или французском языках(желаемое для подбора слово замените на *? ):",
        value="Replace me by *? you'd like. I love *? and my mother, but *? only one who *? me."
    )


def main():
    st.title("")
    
    text = get_text()
    
    unmasker = pipeline('fill-mask', model='camembert-base')
    
    result = split_text[0]
    
    if text != "":
        split_text = text.split("*?")
        
        for i in range(1, len(split_text)):
            tmp_result = unmasker(result + "<mask>" + split_text[i])

            result += tmp_result[0]["token_str"] + split_text[i]

            st.caption("\nВарианты слов и их вероятностное распределение:")
            for el in tmp_result:
                st.write(el["token_str"] + " - " + str(round(el["score"] * 100)) + "%")

        st.caption("\nВозможный финальный вид текста:")
        st.write(result)


if __name__ == "__main__":
    main()
