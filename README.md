# MATH 680 LABS
This is some code that I wrote as a teaching assistant for the class MATH 680 at Texas A&amp;M University (TAMU) for the class's python labs. The labs (by means of this code) demonstrate concrete applications of the concepts presented in lecture, including several exercises. I wrote these as the TA for the class in spring 2024.

---

# Usage
Each of the files in the `labs_680` contains multiple reproducible demonstations, as well as 1-3 exercises. I will walk you through them during our Thursday meetings.

---

# Dependencies

Besides some standard libraries, this repo depends on the folder of code [answers_680](https://github.com/ThomasLastName/answers_680) and, in turn, both depend on the folder of code [quality_of_life](https://github.com/ThomasLastName/quality_of_life). The former's existence is analogous to having exercises in the back of a textbook, rather in the same place where the exercises are stated. The latter is simply a collection of coding shortcuts that I want to be able to use when writing these demos.

---

# Installation Using the Graphical Interface

### Prerequisites for Installation Using the Graphical Interface
- [x] Have python installed and know how to edit run python files
- [x] Know the directory where python is installed on your computer (see below)
- [x] Have the repositories [answers_680](https://github.com/ThomasLastName/answers_680) and [quality_of_life](https://github.com/ThomasLastName/quality_of_life) already stored in your `Lib` folder (those both have their own installation steps similar to the steps for this repo)
- [x] Have the prerequisite standard packages installed:
    - `numpy`, `matplotlib`, and `quality_of_life` for minimal functionality
    - `tensorflow`, `pytorch`, `sklearn` for ~98% functionality
    -  `alive_progress` for complete functionality (this is included for my own sanity when writing the code)

**More on the directory where python is installed:** I recommend having this written down somewhere. You can retrieve this in the interactive python terminal by commanding `import os; import sys; print(os.path.dirname(sys.executable))`. Thus, in Windows, you can probably just open the command line and paste into it `python -c "import os; import sys; print(os.path.dirname(sys.executable))"`. That's the directory where python is located (e.g., `C:\Users\thoma\AppData\Local\Programs\Python\Python310` on my computer) and within *that* directory you can find a folder called `Lib` (e.g., `C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib` on my computer). For reference, this is where many of python's base modules are stored, such as `warnings.py`, `pickle.py`, and `turtle.py`.

### Installation Steps Using the Graphical Interface
Click the colorful `<> Code` button and select `Download ZIP` from the dropdown menu. This should download a zipped folder called `labs_680` containing within it an unzipped folder of the same name, which you just need to click and drag (or copy and paste) into the `Lib` folder of your preferred version of python.

### Subsequent Updates Using the Graphical Interface
You'll have to repeat the process, again. When you attempt to click and drag (or copy and paste) the next time, your operating system probably prompts you with something like "These files already exist! Are you tweaking or did you want to replace them?" and you can just click "replace" or whatever it prompts you with.

---

# Installation Using git

### Prerequisites for Installation Using the git
- [x] All the same prerequisites as for the installation using the graphical interface
- [x] Additionally, have git installed on your computer

### Installation Steps Using the git
Navigate to wherever you want to store these demos on your computer (anywhere is fine), and within that directory command `git clone https://github.com/ThomasLastName/labs_680.git`, which will create a folder called `labs_680` in the same directory.

For instance, if I just want this folder to be on my desktop, then the directory that I want to navigate to would be `C:\Users\thoma\OneDrive\Desktop` on my personal computer. To get there, I would paste `cd C:\Users\thoma\OneDrive\Desktop` into the command line or shell. Then, having already navigated to the directory where I want to create the folder, I am ready to paste `git clone https://github.com/ThomasLastName/labs_680.git` into the command line. That's it!

### Subsequent Updates Using git
Navigate the the directory of the folder that you created, and within that directory command `git pull https://github.com/ThomasLastName/labs_680.git`.

For instance, to continue the example above, if I created the folder `labs_680` on my desktop `C:\Users\thoma\OneDrive\Desktop`, then I'll want to navigate there in the command line by pasting `cd C:\Users\thoma\OneDrive\Desktop\labs_680` and, then, I'm ready to paste `git pull https://github.com/ThomasLastName/labs_680.git` into the command line.
