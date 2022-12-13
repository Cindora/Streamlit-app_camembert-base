from transformers import pipeline
import streamlit as st


def get_text():
    return st.text_area(
        label="Введите текст-маску на английском или французском языках (желаемые для подбора слова замените на *? ):",
        value="Diplomacy is the *? of telling *? to go to *? in such a way that they *?."
    )


@st.cache
def unmask(text):
    split_text = text.split("*?")
    result_text = split_text[0]
    results = [""]
    for i in range(1, len(split_text)):
        unmasker = pipeline('fill-mask', model='camembert-base')
        tmp_result = unmasker(result_text + "<mask>" + split_text[i])

        result_text += tmp_result[0]["token_str"] + split_text[i]

        results.append("")
        for el in tmp_result:
            results[i] += (el["token_str"] + " - " + str(round(el["score"] * 100)) + "%  |  ")

    results[0]=result_text
    return results


def main():
    st.title("")
    text = get_text()

    if bool(st.button('Fill')) & (text != ""):
        results = unmask(text)
        st.caption("\nВарианты слов и их вероятностное распределение:")
        for i in range(1, len(results)):
            st.write(str(i) + ": " + results[i])

        st.caption("Возможный финальный вид текста:")
        st.write(results[0])


if __name__ == "__main__":
    main()
