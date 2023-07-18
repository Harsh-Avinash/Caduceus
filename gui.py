import threading
import psutil
from tkinter import *
from tkinter import scrolledtext

from run import get_answer, main

# Define global variables
qa = None
conversation = []  # Conversation history

# Define colors
bg_color = "#2b2b2b"  # Dark background color
text_color = "#FFFFFF"  # Text color
input_bg_color = "#4b4b4b"  # Input field background color
button_bg_color = "#3e3e3e"  # Button background color
button_fg_color = "#FFFFFF"  # Button text color
processing_color = "#FF0000"  # Color for processing status
ready_color = "#00FF00"  # Color for ready status

# Define font styles
font_family = "Arial"
font_size = 12
header_font_size = 14
font_bold = "bold"

# Define window dimensions
window_width = 800
window_height = 600


def process_query(query):
    # Update processing status box color to red
    status_box.configure(bg=processing_color)

    # Delete "Please wait..." message
    txt.delete('end - 3 lines', 'end - 1 lines')

    # Append current query to conversation history
    conversation.append(query)

    # Join the conversation history with newline characters
    conversation_text = '\n'.join(conversation)

    question, answer, time_taken, docs = get_answer(qa, conversation_text)

    txt.insert(INSERT, f"AI: {answer} (Response time: {time_taken}s)\n\n")
    txt.see("end")

    # Update processing status box color to green
    status_box.configure(bg=ready_color)


def ask_question(event=None):
    query = entry.get()
    entry.delete(0, END)
    if query.strip() != "":
        txt.insert(INSERT, f"User: {query}\n\n")
        txt.insert(INSERT, "AI: Please wait...\n\n")
        txt.see("end")
        threading.Thread(target=process_query, args=(query,)).start()


def update_system_info():
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    system_info_label.configure(text=f"CPU: {cpu_usage}%   RAM: {ram_usage}%")
    window.after(1000, update_system_info)


# Create window
window = Tk()
window.title("AI Chatbot")
window.geometry(f"{window_width}x{window_height}")
window.configure(bg=bg_color)

window.grid_columnconfigure(0, weight=1)
window.grid_rowconfigure(0, weight=1)

# Create text area for chat history
txt = scrolledtext.ScrolledText(window, width=70, height=10, font=(
    font_family, font_size), bg=bg_color, fg=text_color)
txt.grid(column=0, row=0, padx=10, pady=10, sticky="nsew")

# Create frame for user input
input_frame = Frame(window, bg=bg_color)
input_frame.grid(column=0, row=1, padx=10, pady=10, sticky="we")

# Create status box
status_box = Label(input_frame, width=2, bg=ready_color)
status_box.pack(side=LEFT)

# Create entry field for user to type question
entry = Entry(input_frame, font=(font_family, font_size),
              bg=input_bg_color, fg=text_color)
entry.pack(side=LEFT, fill=BOTH, expand=True)
# Allow pressing Enter key to ask question
entry.bind('<Return>', ask_question)

# Create send button
send_button = Button(input_frame, text="Send", command=ask_question, font=(font_family, font_size, font_bold),
                     bg=button_bg_color, fg=button_fg_color, relief=FLAT)
send_button.pack(side=RIGHT, padx=(5, 0))

# Create label for system information
system_info_label = Label(window, text="", font=(
    font_family, font_size), bg=bg_color, fg=text_color)
system_info_label.grid(column=0, row=2, padx=10, pady=10, sticky="w")

# Create label for attribution
attribution_label = Label(window, text="Harsh Avinash 2023", font=(font_family, header_font_size),
                          bg=bg_color, fg=text_color)
attribution_label.grid(column=0, row=3, padx=10, pady=10, sticky="w")

# Load QA model
qa = main()

# Start updating system information
update_system_info()

# Start GUI event loop
window.mainloop()
