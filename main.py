from transformers import pipeline
import streamlit as st


def get_text():
    return st.text_area(
        label="Введите текст-маску на английском или французском языках (желаемые для подбора слова замените на *? ):",
        value="Diplomacy is the *? of telling *? to go to *? in such a way that they *?."
    )


@st.cache
def unmask(text):
    Substring = '[*][?]'
    results = [""]

    menu = re.search(Substring, text)
    if menu:
        result_text = text[:menu.span()[0]]
        index = menu.span()[1]
        menu = re.search(Substring, text[index:])
    else:
        results[0] = text
        return results

    unmasker = pipeline('fill-mask', model='camembert-base')

    while menu:
        tmp_result = unmasker(result_text + "<mask>" + text[index: index + menu.span()[0]])
        result_text += tmp_result[0]["token_str"] + text[index: index + menu.span()[0]]

        results.append("")
        len_list = len(results)-1
        for el in tmp_result:
            results[len_list] += (el["token_str"] + " - " + str(round(el["score"] * 100)) + "%  |  ")

        index += menu.span()[1]

        menu = re.search(Substring, text[index:])
    else:
        tmp_result = unmasker(result_text + "<mask>" + text[index:])
        result_text += tmp_result[0]["token_str"] + text[index:]

        results.append("")
        len_list = len(results)-1
        for el in tmp_result:
            results[len_list] += (el["token_str"] + " - " + str(round(el["score"] * 100)) + "%  |  ")

        results[0] = result_text
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
