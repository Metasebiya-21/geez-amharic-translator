# Compile with XeLaTeX for Ethiopic Unicode
all:
	xelatex -interaction=nonstopmode main.tex
	xelatex -interaction=nonstopmode main.tex

en-fr:
	pdflatex -interaction=nonstopmode en_fr_main.tex
	pdflatex -interaction=nonstopmode en_fr_main.tex

figures:
	python3 generate_figures.py

clean:
	rm -f main.aux main.log main.out main.toc main.pdf
	rm -f en_fr_main.aux en_fr_main.log en_fr_main.out en_fr_main.pdf

.PHONY: all en-fr figures clean
