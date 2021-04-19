.PHONY: watch

default: progress-alex.pdf

progress-alex.pdf: progress-alex.tex progress-alex.bib
	latexmk -bibtex -pdf progress-alex

watch:
	latexmk -bibtex -pdf -pvc progress-alex
