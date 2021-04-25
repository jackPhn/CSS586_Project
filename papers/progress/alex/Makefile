.PHONY: watch

default: progress-alex.pdf

progress-alex.pdf: progress-alex.tex progress-alex.bib
	bibtex progress-alex
	latexmk -bibtex -pdf progress-alex

watch:
	bibtex progress-alex
	latexmk -bibtex -pdf -pvc progress-alex
