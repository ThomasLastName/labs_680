# MATH 680 LABS
This is some code that I wrote as a teaching assistant for the python labs of the class MATH 680 at Texas A&amp;M University. The labs (by means of this code) demonstrate concrete applications of the concepts presented in lecture, and present several optional exercises for students who want to go the extra mile. I wrote these as the TA for the class in spring 2024, typically building on top of the codebase that accompanies the course: [https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition](https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition).


---

# Usage
Each of these files contains multiple reproducible demonstations, as well as 1-3 exercises. I've tried to comment them well enough that they can be read independently. However, I'll also walk you through them during our Thursday meetings at 8:25am in Blocker 129 (in the OAL computer lab).

---

# Prerequisites for Using This Code
Besides some standard libraries, this repo depends on the folder of code [answers_680](https://github.com/ThomasLastName/answers_680) and, in turn, both depend on the folder of code [quality_of_life](https://github.com/ThomasLastName/quality_of_life). The former's existence is analogous to placing the answers to exercises in the back of a textbook, rather in the same place where the exercises are assigned. The latter is simply a collection of coding shortcuts that I want to be able to use when writing these demos, and in every other python project that I engage in.

**List of Requirements in Order to Use this Code:**
- [x] Have python installed and know how to edit and run python files
- [x] Have the repositories [answers_680](https://github.com/ThomasLastName/answers_680) and [quality_of_life](https://github.com/ThomasLastName/quality_of_life) already stored in your `Lib` folder. Those both have their own installation steps, similar to the steps for this repo. See their respective REDAME's for more info
- [x] Have the prerequisite standard packages installed:
    - `numpy`, `matplotlib`, and `quality_of_life` for minimal functionality
    - `tensorflow`, `pytorch`, `sklearn` for ~98% functionality
    -  `alive_progress` for complete functionality (this is included for my own sanity when writing the code)

---

# "Installation" Using Copy+Paste (recommended)

Each week, create a blank `.py` file wherever you prefer, copy the code from this week's `.py` file in GitHub, and paste it into your blank `.py` file.

---

# Installation Using the Graphical Interface (not recommended)

**Installation Steps Using the Graphical Interface:** Click the colorful `<> Code` button at [https://github.com/ThomasLastName/labs_680](https://github.com/ThomasLastName/labs_680) and select `Download ZIP` from the dropdown menu. This should download a zipped folder called `labs_680` containing within it an unzipped folder of the same name, which you just need to click and drag (or copy and paste) to wherever you want to keep these files (anywhere is fine).

**Subsequent Updates Using the Graphical Interface:** You'll have to repeat the process, again. When you attempt to click and drag (or copy and paste) the next time, your operating system probably prompts you with something like "These files already exist! Are you tweaking or did you want to replace them?" and you can just click "replace" or whatever it prompts you with.

**Warning:** Updating the folder in this way will erase any changes you may have made to the `.py` files (e.g., if you filled in the code where it was left blank as an exercise, that will be lost). You can avoid this by renaming the `.py` file in which you are working on the optional exercises.

---

# Installation Using git (not recommended)

**Additional Prerequisites Using git:**
- [x] Have git installed on your computer

**Installation Steps Using git:** In the shell or command line, navigate to wherever you want to store these demos on your computer (anywhere is fine), and within that directory command `git clone https://github.com/ThomasLastName/labs_680.git`, which will create and populate a folder called `labs_680` in the same directory.

For example, if I just want this folder to be on my desktop, then the directory that I need to navigate to would be `C:\Users\thoma\OneDrive\Desktop` on my personal computer. To get there, I would paste `cd C:\Users\thoma\OneDrive\Desktop` into the Windows command line. Then, having navigated to the directory where I want to create the folder, I am ready to paste `git clone https://github.com/ThomasLastName/labs_680.git` into the command line. That's it!

**Subsequent Updates Using git:**
Navigate to the directory of the folder that you created, and within that directory command `git pull https://github.com/ThomasLastName/labs_680.git`.

For instance, to continue the example above, if I created the folder `labs_680` on my desktop `C:\Users\thoma\OneDrive\Desktop`, then I'll want to navigate there by pasting `cd C:\Users\thoma\OneDrive\Desktop\labs_680` into the Windows command line. Next, I paste `git pull https://github.com/ThomasLastName/labs_680.git` into the command line.

**Warning:** Updating the folder in this way will erase any changes you may have made to the `.py` files (e.g., if you filled in the code where it was left blank as an exercise, that will be lost). You can avoid this by renaming the `.py` file in which you are working on the optional exercises.
