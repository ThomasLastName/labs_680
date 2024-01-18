# MATH 680 LABS
This is some code that I wrote for the python labs of the class MATH 680 at Texas A&amp;M University. The labs (by means of this code) demonstrate concrete applications of the concepts presented in lecture, and present several optional exercises for students who want to go the extra mile. I wrote these as the teaching assistant for the course in spring 2024, typically building on top of the codebase that accompanies the course: [https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition](https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition).

---

# Usage
Each of these files contains multiple reproducible demonstations, as well as 1-3 exercises. I've tried to comment them well enough that they can be read independently. However, I'll also walk you through them during our Thursday meetings at 8:25am in Blocker 129 (in the OAL computer lab).

In order to participate in the optional exercise(s), switch from `exercise_mode = False` to `exercise_mode = True` before the import statements. The code will then prompt errors, unless of course you complete the exercise(s), which you can locate by Crtl+F-ing either "exercise_mode" or (case sensitively) "EXERCISE".

---

# Prerequisites for Using This Code
Besides some standard libraries, this repo depends on the folder of code [answers_680](https://github.com/ThomasLastName/answers_680) and, in turn, both depend on the folder of code [quality_of_life](https://github.com/ThomasLastName/quality_of_life). The former's existence is analogous to placing the answers to exercises in the back of a textbook, rather in the same place where the exercises are assigned. The latter is simply a collection of coding shortcuts that I want to emply when writing these demos, just to make my own life easier. See the Installation, below.

**List of Requirements in Order to Use this Code:**
- [x] Have python installed and know how to edit and run python files
- [x] Have the prerequisite standard packages installed:
    - `numpy` and `matplotlib` for minimal functionality
    - `tensorflow`, `pytorch`, `sklearn` for ~98% functionality
    -  `alive_progress` is kind of optional (only used in 1 or 2 examples; will not provoke an error)

---

# Installation

---

## Installation Using Copy+Paste (recommended)

Each week, create a blank `.py` file wherever you prefer, copy the code from this week's `.py` file in GitHub, and paste it into your blank `.py` file. Then, modify the code at the following two points:
 - Replace `install_assist = False` with `install_assist = True`
 - Replace `confirm_permission_to_modify_files = not install_assist` with `confirm_permission_to_modify_files = True`

If these two changes are implemented, then the code will automatically download various `.py` files from [answers_680](https://github.com/ThomasLastName/answers_680) and [quality_of_life](https://github.com/ThomasLastName/quality_of_life) each time it is executed, including earsing and replacing those files (effectively updating them) if they were already present. In other words, the code will effectively follow the [Installation Using the Graphical Interface](https://github.com/ThomasLastName/labs_680?tab=readme-ov-file#installation-using-the-graphical-interface-not-recommended) instuctions every time it is executed.

**Disclaimer:** By implementing those two changes in your code, you are consenting to me modifying the files on your computer, with the understanding that it is possible I will make a mistake.

**Warning:** If you _coincidentally_ have files which you don't want to lose installed in the same place as the donwloads are targetted to, and if your file has the same name as the one being downloaded, then your file will be erased and replaced by the downloaded one. To see the names of the file Ctrl+f `files = [`, which should appear in two separate places a few lines apart.

---

## Installation Using the Graphical Interface (not recommended)

**Additional Prerequisites Using the Graphical Interface:**
- [x] Have the repositories [answers_680](https://github.com/ThomasLastName/answers_680) and [quality_of_life](https://github.com/ThomasLastName/quality_of_life) already stored in your `Lib` folder. Those both have their own installation steps, similar to the steps for this repo. See their respective README's for more info

**Installation Steps Using the Graphical Interface:** Click the colorful `<> Code` button at [https://github.com/ThomasLastName/labs_680](https://github.com/ThomasLastName/labs_680) and select `Download ZIP` from the dropdown menu. This should download a zipped folder called `labs_680` containing within it an unzipped folder of the same name, which you just need to click and drag (or copy and paste) to wherever you want to keep these files (anywhere is fine).

**Subsequent Updates Using the Graphical Interface:** You'll have to repeat the process, again. When you attempt to click and drag (or copy and paste) the next time, your operating system probably prompts you with something like "These files already exist! Are you tweaking or did you want to replace them?" and you can just click "replace" or whatever it prompts you with.

**Warning:** Updating the folder in this way will erase any changes you may have made to the `.py` files (e.g., if you filled in the code where it was left blank as an exercise, that will be lost). You can avoid this by renaming the `.py` file in which you are working on the optional exercises.

---

## Installation Using git

**Additional Prerequisites Using git:**
- [x] Have the repositories [answers_680](https://github.com/ThomasLastName/answers_680) and [quality_of_life](https://github.com/ThomasLastName/quality_of_life) already stored in your `Lib` folder. Those both have their own installation steps, similar to the steps for this repo. See their respective README's for more info
- [x] Have git installed on your computer

**Installation Steps Using git:** In the shell or command line, navigate to wherever you want to store these demos on your computer (anywhere is fine), and within that directory command `git clone https://github.com/ThomasLastName/labs_680.git`, which will create and populate a folder called `labs_680` in the same directory.

For example, if I just want this folder to be on my desktop, then the directory that I need to navigate to would be `C:\Users\thoma\OneDrive\Desktop` on my personal computer. To get there, I would paste `cd C:\Users\thoma\OneDrive\Desktop` into the Windows command line. Then, having navigated to the directory where I want to create the folder, I am ready to paste `git clone https://github.com/ThomasLastName/labs_680.git` into the command line. That's it!

**Subsequent Updates Using git:**
Navigate to the directory of the folder that you created, and within that directory command `git pull https://github.com/ThomasLastName/labs_680.git`.

For instance, to continue the example above, if I created the folder `labs_680` on my desktop `C:\Users\thoma\OneDrive\Desktop`, then I'll want to navigate there by pasting `cd C:\Users\thoma\OneDrive\Desktop\labs_680` into the Windows command line. Next, I paste `git pull https://github.com/ThomasLastName/labs_680.git` into the command line.

**Warning:** Updating the folder in this way will erase any changes you may have made to the `.py` files (e.g., if you filled in the code where it was left blank as an exercise, that will be lost). You can avoid this by renaming the `.py` file in which you are working on the optional exercises.
