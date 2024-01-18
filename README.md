# MATH 680 LABS
This is some code that I wrote for the python labs of the class MATH 680 at Texas A&amp;M University. The labs (by means of this code) demonstrate concrete applications of the concepts presented in lecture, and present several optional exercises for students who want to go the extra mile. I wrote these as the teaching assistant for the course in spring 2024, typically building on top of the codebase that accompanies the course: [https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition](https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition).

---

# Usage
Each of these files contains multiple reproducible demonstations, as well as 1-3 exercises. I've tried to comment them well enough that they can be read independently. However, I'll also walk you through them during our Thursday meetings at 8:25am in Blocker 129 (in the OAL computer lab).

In order to participate in the optional exercise(s), switch from `exercise_mode = False` to `exercise_mode = True` before the import statements. The code will then prompt errors, unless of course you complete the exercise(s), which you can locate by Crtl+F-ing either "exercise_mode" or (case sensitively) "EXERCISE".

---

# Prerequisites for Using This Code
Besides some standard libraries, this repo depends on the folder of code [answers_680](https://github.com/ThomasLastName/answers_680) and, in turn, both depend on the folder of code [quality_of_life](https://github.com/ThomasLastName/quality_of_life). The former is analogous to placing the answers to exercises in the back of a textbook, rather in the same place where the exercises are assigned. The latter is simply a collection of coding shortcuts that I want to have access to when writing these demos, to make my own life easier. **See [Installation](https://github.com/ThomasLastName/labs_680?tab=readme-ov-file#installation), below.**

**List of Requirements in Order to Use this Code:**
- [x] Have python installed and know how to edit and run python files
- [x] Have the prerequisite standard packages installed:
    - `numpy` and `matplotlib` for minimal functionality
    - `tensorflow`, `pytorch`, `sklearn` for ~98% functionality
    -  `alive_progress` is kind of optional (only used in 1 or 2 examples; will not provoke an error)

---

# Installation

I've made an effort to write this code such that it, if you simply Copy+Paste it into a Colab notebook, then it will "magically just work." Although, if you do so, then it's your prerogative to split the code into cells.

More generally, in order for the `.py` files in this repo to run correctly, you need to download the `.py` files from [answers_680](https://github.com/ThomasLastName/answers_680) and [quality_of_life](https://github.com/ThomasLastName/quality_of_life), put them in respective folders of the same name, and put those folders on the path: i.e., put those folders in a direcory such that import statements like `from quality_of_life import ansi` work. There are a variety of places you can put the folders such that this will work, depending on your Python environment.

**Advanced Users:** You can do this yourself by git cloning this as well as the aforementioned two repos to the location of your preference.
 
**All Users:** I recommend following the [Installation Using Copy+Paste](https://github.com/ThomasLastName/labs_680?tab=readme-ov-file#installation-using-copypaste-recommended) instructions, below.

---

## Installation Using Copy+Paste (recommended)

Each week, create a blank Python file (or notebook) wherever you prefer (e.g., Colab or IDLE). Copy the code from this week's `.py` file in GitHub, and paste it into your blank Python file. Then, modify the code at the following two points:
 - Replace `install_assist = False` with `install_assist = True`
 - Replace `confirm_permission_to_modify_files = not install_assist` with `confirm_permission_to_modify_files = True`

If these two changes are implemented, then the code will automatically download various `.py` files from [answers_680](https://github.com/ThomasLastName/answers_680) and [quality_of_life](https://github.com/ThomasLastName/quality_of_life) and attempt to locate them in an adequate directory each time it is executed, erasing and replacing those files (effectively updating them) if they were already present.

More precisely, if the above two changes are implemented, then the code will look for folders called `answers_680` and `quality_of_life` within the directory where your Python environment stores packages (as determined by `os.path.dirname(os.path.dirname(np.__file__))`), it will create those two folders if they don't already exist, and it will download a variety of `.py` files and place them inside of those folders, erasing the ones of the same name that were previously there (if any).  To see the names of the files that the code will download if the above two changes are implemented, you can Ctrl+f `files = [`, which should appear in two separate places.

**Disclaimer:** By implementing those two changes in your code, you are consenting to me modifying the files on your computer, with the understanding that it is possible I will make a mistake. For example, if you _coincidentally_ have files _which you don't want to lose_ installed in the same place as the donwloads are targetted to, and if your files _coincidentally_ have the same name as the one being downloaded, then your files will be erased and replaced by the downloaded ones. However, I think this is highly unlikely.

**Warning:** I have tried to make the `install_assist` block of code as general as possible, but I cannot anticipate all possible environments, and it may not work as intended on your machine. I have made an effort to ensure that this code works as intended in `Colab`, where there is also no risk some unforseen circumstance messing up the Python environment on your computer.
