# def modify_command(Command):
#     print("Do you want to modify the instruction?")
#     Ask = input("\n(Enter 0 for No, 1 for Yes): ")

#     if Ask == "1":
#         Command = input("Enter the new prompt:")
#         return Command
#     elif Ask == "0":
#         return Command
#     else:
#         print("Invalid input. Please enter 0 or 1.")
#         return modify_command(Command)
    
# def get_instruction():
#     print("Do you want to provide the instruction?")
#     Ask = input("\n(Enter 0 for No, 1 for Yes): ")
#     instr=''
#     if Ask == "1":
#         instr = input("Enter the instruction:")
#         return "I: "+instr+"\n"
#     elif Ask == "0":
#         return instr
#     else:
#         print("Invalid input. Please enter 0 or 1.")
#         return get_instruction()

# def get_context():
#     print("Do you want to provide the context?")
#     Ask = input("\n(Enter 0 for No, 1 for Yes): ")
#     context=''
#     if Ask == "1":
#         context = input("Enter the context:")
        
#         return "C: "+context+"\n"
#     elif Ask == "0":
#         return context
#     else:
#         print("Invalid input. Please enter 0 or 1.")
#         return get_context()

# class prompt_suggest:
#     def __init__(self):
#         # Do myself/ self จะแทนที่ตัวแปร
#         self.name = 'Chommakorn'
#         self.lastname = 'Sontesadisai'
#         self.nickname = 'Kafka'

#     def std_sent(template: str):
#         """
#         ...
        
        
#         """
#         # Instruction of Standard Prompting
#         Command = "Classify the sentiment of the text: "
#         new_command = modify_command(Command)
#         Command = new_command

#         if template == 'Zero Shot':
#             Text = input("Enter the text for sentiment analysis: ")
#             Expected_output = input("Enter the expected output: ")

#             Prompt = Command+"\n\nText: "+Text+"\nSentiment:"
#             return Prompt, Expected_output
#         elif template == 'One Shot':
#             Text = input("Enter the text example: ")
#             Sentiment = input("Enter the answer for the example: ")

#             Text_2 = input("Enter the text example: ")
#             Prompt = f"{Command}\n\nExample 1\nText: \"{Text}\"\nSentiment: {Sentiment}\n\nText: {Text_2}\nSentiment:"
           
#             Expected_output = input("Enter the expected output: ")
#             return Prompt, Expected_output
#         elif template == 'Few Shot':
#             count = int(input("Enter the number of examples: "))
#             Text = ''
#             for i in range(count):
#                 example_text = input(f"Enter text for example {i+1}: ")
#                 example_sentiment = input(f"Enter sentiment for example {i+1} : ")
#                 Text += f"Text: \"{example_text}\"\nSentiment: {example_sentiment}\n\n"
#             Text_2 = input("Enter the text example: ")
#             Expected_output = input("Enter the expected output: ")

#             Prompt = Command+"\n\n"+Text + f"Text: {Text_2} \nSentiment:"
#             return Prompt, Expected_output
        

#     def std_qna(template: str):
#         Instr = get_instruction()
#         Context = get_context()

#         if template == 'Zero Shot':
#             Q = input("Enter the Question: ")
#             Expected_output = input("Enter the Expected Output: ")

#             Prompt = f"""{Instr}\n{Context}Q: {Q}\nA:"""
#             return Prompt, Expected_output

#         elif template == 'One Shot':
#             # One example input from the user
#             Q_1 = input("Enter the Question #1: ")
#             A_1 = input("Enter the Answer #1: ")

#             Q = input("Enter the Question: ")
#             Prompt = f"{Instr}\n{Context}\nQ: \"{Q_1}\"\nA: {A_1}\n\nQ: {Q}\nA:"
#             Expected_output = input("Enter the expected output: ")

#             return Prompt, Expected_output

#         elif template == 'Few Shot':
#             count = int(input("Enter the number of examples: "))
#             Text = ''
#             for i in range(count):
#                 example_text = input(f"Enter the Question #{i+1}: ")
#                 example_sentiment = input(f"Enter the Answer #{i+1} : ")
#                 Text += f"Q: \"{example_text}\"\nA: {example_sentiment}\n\n"

#             Q = input("Enter the Question: ")
#             A = input("Enter the Expected Output: ")

#             Prompt = f"""{Instr}\n{Context}\n{Text}Q: {Q} \nA:"""
#             return Prompt, Expected_output

#     def cot_sent(template: str):
#         return ''
    
#     def cot_qna(template: str):
#         return ''

    


#     # dependecies of ,y libr installation
#     # get template
#     # name
#     # user need to provide api/get completion function แยก
#     # แยก module ยังไงให้ใช้งานง่าย??

# myjublek = prompt_suggest()
# print(myjublek.name)

# # import prompt_suggest
# # import prompt_scoring