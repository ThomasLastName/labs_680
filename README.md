# MATH 680 LABS
This is some code that I wrote for the python labs of the class MATH 680 at Texas A&amp;M University. The labs (by means of this code) demonstrate concrete applications of the concepts presented in lecture, and present several optional programming exercises. I wrote these as the teaching assistant for the course in spring 2024, typically building on top of [the official codebase for the text](https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition).

---

# Usage
Each of these files contains multiple reproducible demonstations, as well as 1 or more programming exercises for students seeking more programming experience. I've tried to comment them well enough that they can be read independently. However, I'll also walk you through them during our **Tuesday meetings at 5:30pm in Blocker 122.**

**In order to participate in the optional exercise(s), switch from `exercise_mode = False` to `exercise_mode = True`** before the import statements. The code will then prompt errors, unless of course you complete the exercise(s), which you can locate by Crtl+F-ing either "exercise_mode" or (case sensitively) "EXERCISE". You may find the answer key [here](https://github.com/ThomasLastName/answers-680).

**For a user-friendly experience, I've packaged the code in colab notebooks, accessible [HERE](https://drive.google.com/drive/folders/1rhDQxKEpeTkqFpTRY6NqJma0naKqD4Nd?usp=drive_link)**  with a Google sign in.

---

# Prerequisites for Using This Code
Besides some standard libraries, this repo depends two packaes that I wrote: [answers-680](https://github.com/ThomasLastName/answers-680) and [quality-of-life](https://github.com/ThomasLastName/quality-of-life). The former is analogous to placing the answers to exercises in the back of a textbook, rather in the same place where the exercises are assigned. The latter is simply a collection of coding shortcuts that I want to have access to when writing these demos, to make my own life easier. **See [Installing Tom's Packages](https://github.com/ThomasLastName/labs_680?tab=readme-ov-file#installation), below.**

**List of Requirements in Order to Use this Code:**
1. Have python installed and know how to edit and run python files.
2. Have [answers-680](https://github.com/ThomasLastName/answers-680) and [quality-of-life](https://github.com/ThomasLastName/quality-of-life) installed.
3. Have the prerequisite standard packages installed:
    * `numpy` and `matplotlib` for minimal functionality
    * `torch` as well, for the PyTorch exercises in later weeks
    * `scipy`, `plotly`, `cvxpy`, `ecos`, `tensorflow`, `scikit-learn` for complete functionality (I can't actually remember whether or not `scipy` and `plotly` are indeed necessary; so, I'm including them to be on the safe side)

---

# Installing Tom's Packages

## Installation in Colab (beginner friendly)

I've made an effort to write this code such that it, if you Copy+Paste it into a Colab notebook, it will magically just work. Although, if you do so, then it's your prerogative to split the code into cells. As allueded to above, I've done this for you. See [usage](https://github.com/ThomasLastName/labs_680?tab=readme-ov-file#usage).


## Installation in Any Environment Using `git` (requires `git`)

`pip install --upgrade git+https://github.com/ThomasLastName/answers-680.git` should take care of everything for you. If the code doesn't work after that, please tell me and/or submit an [issue](https://github.com/ThomasLastName/labs_680/issues)!

---

## Assisted Installation for Environments other than Colab (deprecated)

Each week, create a blank Python file (or notebook) wherever you prefer. Copy the code from this week's `.py` file in GitHub, and paste it into your blank Python file. Then, modify the code at the following two points:
1. Replace `install_assist = False` with `install_assist = True`
2. Replace `confirm_permission_to_modify_files = not install_assist` with `confirm_permission_to_modify_files = True`

That's it! If these two changes are implemented, then the code will automatically download various `.py` files from the deprecated repositories [answers_680](https://github.com/ThomasLastName/answers_680) and [quality_of_life](https://github.com/ThomasLastName/quality_of_life) and _attempt_ to locate them in an adequate directory each time it is executed, erasing and replacing those files (effectively updating them) if they were already present.

More precisely, if the above two changes are implemented, then the code will look for folders called `answers_680` and `quality_of_life` within the directory where your Python environment stores packages (as determined by `os.path.dirname(os.path.dirname(np.__file__))`), it will create those two folders if they don't already exist, and it will download a variety of `.py` files and place them inside of those folders, erasing the ones of the same name that were previously there (if any).  To see the names of the files that the code will download if the above two changes are implemented, you can Ctrl+f `files = [`, which should appear in two separate places.

**Disclaimer:** By implementing those two changes in your code, you are consenting to me modifying the files on your computer, with the understanding that it is possible I will make a mistake. For example, if you _coincidentally_ have files _which you don't want to lose_ installed in the same place as the donwloads are targetted to, and if one your files _coincidentally_ has the same name as one of the ones being downloaded, that file will be erased and replaced by the downloaded one. However, I think this is extraordinarily unlikely.

**Warning:** I have tried to make the `install_assist` block of code as general as possible, but I cannot anticipate all possible environments, and it may not work as intended on your machine. I have made an effort to ensure that this code works as intended in `Colab`, where there is also no risk some unforseen circumstance messing up the Python environment on your computer.
