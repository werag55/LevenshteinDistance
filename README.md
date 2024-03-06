# LevenshteinDistance

**Overview:**

The program is used to determine the Levenshtein edit distance between two strings of characters consisting of letters of the Latin alphabet. Additionally, the program returns the path of transformations needed to transform the first of the input strings into the second.

The project aimed to compare the performance of two algorithms - sequential (dynamic programming) and parallel (CUDA).

**Instruction**

We can run the program with the following options:
  - -h, --help Displays user instructions.
  - -c, --cuda Runs only the algorithm using CUDA.
  - -sq, --sequential Runs only the sequential algorithm.
    Note: If we want to run both algorithms, we do not use the -c and -sq options - these options are for running only one selected algorithm and cannot be used simultaneously.
  - -tm, --time More detailed time measurements will be displayed (without this option only measurements of the execution time of entire algorithms).
  - -s WORD, --source WORD Specify the source word from the console.
  - -t WORD, --target WORD Specify the target word from the console.
  - -f FILE, --file FILE Specify the name of the file with the words that are the input of the algorithm - the source word in the first line, the target word in the second.
                         The file should be in the same folder as the .exe file
                         If the option is used together with -s or -t, only -f will be taken into account.
  - -hd, --hide If used, the source word, target word, and the transformation path between them will not be displayed on the console (for large tests).

**Path of transformation - markings**
- \- - leaving the character unchanged
- d - removing a character from the source word
- i - insert a character from the target word
- s - sign replacement ( = d + i)

e.g. let s = karnisz, t = dokarmia, then the transformation path is ii---s-ds, i.e.:
Let's set the "pointers" p to the first letter of the word s and q to the first letter of the word t
- i - insert the letter 'd' (pointed to by q) from t, move one character to the right with the q pointer (currently we have the word "d")
- i - insert the letter 'o' (pointed to by q) from t, move one character to the right with the q pointer (currently we have the word "to")
- \- - we leave the letter 'k' (pointed to by p) from s, move one character to the right with pointers p and q (currently we have the word "dock")
- \- - we leave the letter 'a' (pointed to by p) from s, move one character to the right with pointers p and q (currently we have the word "doka")
- \- - we leave the letter 'r' (pointed to by p) from s, move one character to the right with pointers p and q (currently we have the word "dokar")
- s - we replace the letter 'n' (pointed by p) from s with the letter 'm' (pointed by q) from t, we move one character to the right with pointers p and q (currently we have the word "dokarm")
- \- - we leave the letter 'i' (pointed to by p) from s, move one character to the right with pointers p and q (currently we have the word "dokarmi")
- d - remove the letter 's' (pointed to by p) from s, move one character to the right with the p pointer (currently we have the word "dokarmi")
- s - we replace the letter 'z' (pointed to by p) from s with the letter 'a' (pointed to by q) from t, we move one character to the right with pointers p and q ( -> end of words) (currently we have the word "dokarmia")
