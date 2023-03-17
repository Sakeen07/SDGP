from tkinter import *
import googletrans
import textblob
from tkinter import ttk, messagebox

root = Tk()
root.title = "Translator"
root.iconbitmap()  # file path
root.geometry("880x300")


def translate_it():
    # Deleting Any existing words
    translated_text.delete(1.0, END)
    try:
        for key, value in languages.items():
            if value == original_combo.get():
                from_language_key = key

        for key, value in languages.items():
            if value == translated_combo.get():
                to_language_key = key

        # Turn Original text in to textblob
        words = textblob.TextBlob(original_text.get(1.0, END))

        # Translate Text
        words = words.translate(from_lang=from_language_key, to=to_language_key)

        # Output Text
        translated_text.insert(1.0, words)


    except Exception as e:
        messagebox.showerror("Translator", e)


def clear():
    original_text.delete(1.0, END)
    translated_text.delete(1.0, END)


# Adding Language list from googleTrans
languages = googletrans.LANGUAGES
language_list = list(languages.values())

# Creating a Text box for original text field
original_text = Text(root, height=10, width=45)
original_text.grid(row=0, column=0, pady=20, padx=10)

# Translate button
translate_button = Button(root, text="Translate ", font=("Helvetica", 20), command=translate_it)
translate_button.grid(row=0, column=1, padx=10)

# Creating a Text box for translated text field
translated_text = Text(root, height=10, width=50)
translated_text.grid(row=0, column=2, pady=20, padx=10)

# Creating Combo Boxes
original_combo = ttk.Combobox(root, width=50, value=language_list)
original_combo.current(21)
original_combo.grid(row=1, column=0)

translated_combo = ttk.Combobox(root, width=50, value=language_list)
translated_combo.current(26)
translated_combo.grid(row=1, column=2)

# Creating A clear button
clear_button = Button(root, text="Clear", command=clear)
clear_button.grid(row=2, column=1)

root.mainloop()
