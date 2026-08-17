--------------------------------------------------------------------------------

You can use Overleaf (https://www.overleaf.com) as an online LaTeX editor.
Directly import this ZIP file using Overleaf interface.

--------------------------------------------------------------------------------

Package Structure

->  ZIP-FILE
        |-> chapters
        |       |-> 1.introduction.tex
        |       |...
        |       |-> appendices.tex
        |
        |-> figures
        |       |...
        |
        |-> frontmatter
        |       |-> abstract-en.tex
        |       |-> abstract-tr.tex
        |       |-> acknowledgements.tex
        |       |-> foreword.tex
        |       |-> preface.tex
        |       |-> symbols.tex
        |
        |-> main.tex
        |-> README (this file)
        |-> references.bib
        |-> thesis.cls
        |-> vancouver.bst

* main.tex
    This file contains the general thesis outline and includes the chapter
    files defined in "chapters" folder according to their expected ordering.

    You should also define your title, name, surname, department, supervisor,
    co-supervisor(s) and examiners at the beginning of this file.

    You can add necessary chapters as needed to your thesis, while keeping in
    mind the FBE thesis requirements.

* thesis.cls
    This file defines the thesis class. It provides you with definitions
    necessary to typeset a thesis in the format required by the Graduate School
    of Natural and Applied Sciences, Yeditepe University.

    Please DO NOT MODIFY this file.

* references.bib
    Add your references using BibTeX format.

* vancouver.bst
    Vancouver style definitions for references.

    Please DO NOT MODIFY this file.

* chapters
    This folder contains the thesis chapters which should be included in the
    main document body.

    * 1.introduction.tex
        Write your introduction chapter in this file.

    * 2.instructions.tex to 5.publication.tex
        Write your chapters that should be between introduction and conclusions
        in separate files as defined in this thesis latex example.

        You can add necessary chapters in between. You can rename these files
        also, but don't forget to change them in main file as well.

    * 6.conclusions.tex
        Write your conclusions chapter in this file.

    * appendices.tex
        Write your appendices in this file.

* figures
    This folder contains all of the figures that should be used in the thesis.

* frontmatter
    This folder contains the thesis frontmatter such as acknowledgements,
    abstract, symbols and abbreviations, etc.

    * abstract-en.tex
        Write your abstract in this file.

    * abstract-tr.tex
        Write your turkish abstract (özet) in this file.

    * acknowledgements.tex
        Write your acknowledgements in this file.

    * foreword.tex
        Write your foreword in this file.

    * preface.tex
        Write your preface in this file.
        Note that if your thesis includes a preface, then you shouldn't include
        an extra acknowledgements page. Write your acknowledgements also in
        this file.

    * symbols.tex
        Define your symbols and abbreviations in this file.

--------------------------------------------------------------------------------

Sectioning Commands

- Main Headings         -> \chapter{Main Heading Name}
- Second Headings       -> \section{Second Heading Name}
- First Subheadings     -> \subsection{First Subheading Name}
- Second Subheadings    -> \subsubsection{Second Subheading Name}
- Third Subheadings     -> \paragraph{Third Subheading Name}

--------------------------------------------------------------------------------
