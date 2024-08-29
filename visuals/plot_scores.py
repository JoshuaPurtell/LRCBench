import matplotlib.pyplot as plt


## Coding Problem

# Data
y_claude_3_opus = [7, 7, 7, 6, 6, 4, 5]
y_claude_3_5_sonnet = [8, 7, 6, 5, 5]
y_gpt_4_32k = [7, 6, 6, 5, 5]
y_gemini_1_5_pro = [6, 6, 5, 5]
y_gemini_1_5_flash = [7, 6, 5, 4]
y_gpt_4o_2024_08_06 = [6, 5, 5]
y_gpt_4o_mini = [7, 4, 4]
y_gpt_4_turbo = [5]

# Updated x values
x_values = [5, 10, 20, 30, 40, 50, 60]

x_claude_3_opus = x_values[:len(y_claude_3_opus)]
x_claude_3_5_sonnet = x_values[:len(y_claude_3_5_sonnet)]
x_gpt_4_32k = x_values[:len(y_gpt_4_32k)]
x_gemini_1_5_pro = x_values[:len(y_gemini_1_5_pro)]
x_gemini_1_5_flash = x_values[:len(y_gemini_1_5_flash)]
x_gpt_4o_2024_08_06 = x_values[:len(y_gpt_4o_2024_08_06)]
x_gpt_4o_mini = x_values[:len(y_gpt_4o_mini)]
x_gpt_4_turbo = x_values[:len(y_gpt_4_turbo)]

plt.figure(figsize=(10, 6))
plt.plot(x_claude_3_opus, y_claude_3_opus, marker='o', linestyle='-', label='Claude-3-opus-20240229')
plt.plot(x_claude_3_5_sonnet, y_claude_3_5_sonnet, marker='o', linestyle='-', label='Claude-3-5-sonnet-20240620')
plt.plot(x_gpt_4_32k, y_gpt_4_32k, marker='o', linestyle='-', label='GPT-4-32k')
plt.plot(x_gemini_1_5_pro, y_gemini_1_5_pro, marker='o', linestyle='-', label='Gemini-1.5-pro')
plt.plot(x_gemini_1_5_flash, y_gemini_1_5_flash, marker='o', linestyle='-', label='Gemini-1.5-flash')
plt.plot(x_gpt_4o_2024_08_06, y_gpt_4o_2024_08_06, marker='o', linestyle='-', label='GPT-4o-2024-08-06')
plt.plot(x_gpt_4o_mini, y_gpt_4o_mini, marker='o', linestyle='-', label='GPT-4o-mini')
plt.plot(x_gpt_4_turbo, y_gpt_4_turbo, marker='o', linestyle='-', label='GPT-4-turbo')

# Adding labels and title
plt.xlabel('N Helper Functions to Choose From')
plt.ylabel('N Questions Answered Correctly')
plt.title('Correctness of Models on Coding Problem')
plt.legend()
plt.grid(True)

# Show plot
plt.savefig("visuals/coding_problem.png")


## Transactions Problem

# Transactions Problem

# Data
y_claude_3_5_sonnet = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 8, 10, 7, 8, 6, 3, 8, 5, 2]
y_gemini_1_5_pro    = [10, 10, 10, 7, 8, 8, 9, 9, 8, 7, 7, 7, 8, 3, 8, 4, 4]
y_claude_3_opus = [9, 10, 10, 8, 9, 10, 10, 9, 8, 7, 8, 7, 7, 2, 3]
y_gpt_4_32k = [10, 10, 9, 8, 9, 8, 7, 9, 6, 6, 9, 5, 5]
y_gemini_1_5_flash = [10, 10, 6, 6, 2, 3]
y_gpt_4_turbo = [10, 10, 7, 6, 5, 2]
y_gpt_4o_2024_08_06 = [10, 10, 8, 2, 3]
y_gpt_4o_mini = [10, 8, 5, 3]
y_gemini_1_5_flash_8b_exp_0827 = [10, 4, 2]

# Updated x values
x_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 400, 500]

x_claude_3_5_sonnet = x_values[:len(y_claude_3_5_sonnet)]
x_gemini_1_5_pro = x_values[:len(y_gemini_1_5_pro)]
x_claude_3_opus = x_values[:len(y_claude_3_opus)]
x_gpt_4_32k = x_values[:len(y_gpt_4_32k)]
x_gemini_1_5_flash = x_values[:len(y_gemini_1_5_flash)]
x_gpt_4_turbo = x_values[:len(y_gpt_4_turbo)]
x_gpt_4o_2024_08_06 = x_values[:len(y_gpt_4o_2024_08_06)]
x_gpt_4o_mini = x_values[:len(y_gpt_4o_mini)]
x_gemini_1_5_flash_8b_exp_0827 = x_values[:len(y_gemini_1_5_flash_8b_exp_0827)]

plt.figure(figsize=(10, 6))
plt.plot(x_claude_3_5_sonnet, y_claude_3_5_sonnet, marker='o', linestyle='-', label='Claude-3-5-sonnet-20240620')
plt.plot(x_gemini_1_5_pro, y_gemini_1_5_pro, marker='o', linestyle='-', label='Gemini-1.5-pro')
plt.plot(x_claude_3_opus, y_claude_3_opus, marker='o', linestyle='-', label='Claude-3-opus-20240229')
plt.plot(x_gpt_4_32k, y_gpt_4_32k, marker='o', linestyle='-', label='GPT-4-32k')
plt.plot(x_gemini_1_5_flash, y_gemini_1_5_flash, marker='o', linestyle='-', label='Gemini-1.5-flash')
plt.plot(x_gpt_4_turbo, y_gpt_4_turbo, marker='o', linestyle='-', label='GPT-4-turbo')
plt.plot(x_gpt_4o_2024_08_06, y_gpt_4o_2024_08_06, marker='o', linestyle='-', label='GPT-4o-2024-08-06')
plt.plot(x_gpt_4o_mini, y_gpt_4o_mini, marker='o', linestyle='-', label='GPT-4o-mini')
plt.plot(x_gemini_1_5_flash_8b_exp_0827, y_gemini_1_5_flash_8b_exp_0827, marker='o', linestyle='-', label='Gemini-1.5-flash-8b-exp-0827')

plt.xlabel('N Transactions to Choose From')
plt.ylabel('N Questions Answered Correctly')
plt.title('Correctness of Models on Transactions Problem')
plt.legend()
plt.grid(True)

plt.savefig("visuals/transactions_problem.png")
