#!/bin/bash
mkdir gray_letters
cp helveticaneue.ttf gray_letters
cp helveticaneue-light-001.ttf gray_letters

graytone=10.0
scriptwidth="375pt"
scriptheight="667pt"
dpidensity=326

for Y in {1..10}
do
  #Create random sequence
  rseq=""
  for i in {1..40}
  do
		for j in {1..40}
		do
    	rseq="$rseq$(($RANDOM%10))"
  	done
		rseq=$(echo "$rseq\\\\\\\\ \\n\t")
	done
	rseqjoined=$(echo "$rseq" | sed "s/-/ \\& /g")
	echo $rseqjoined

  sed -e "s/graytone/$graytone/g; s/scriptwidth/$scriptwidth/; s/scriptheight/$scriptheight/; s/\&/$rseqjoined/I" xetexMakeSMS_gray.tex > gray_letters/gridDigit$Y.tex
  rm gray_letters/tmp.tex

	#remove non-digits to save as text file
	rseq=$(echo $rseq | sed s/'hline'//g | sed s/'n'//g | sed s/'t'//g | sed s/' '//g | sed s/'\\'//g | sed s/'-'//g)

  cd gray_letters
	echo $rseq > gridDigit$Y.txt
  xelatex -synctex=1 -interaction=nonstopmode gridDigit$Y.tex
  convert -antialias -density $dpidensity -quality 100 gridDigit$Y.pdf gridDigit$Y.jpg
  cd ..
  rm gray_letters/gridDigit$Y.tex
  rm gray_letters/gridDigit$Y.log
  rm gray_letters/gridDigit$Y.aux
  rm gray_letters/gridDigit$Y.synctex.gz
done

rm gray_letters/helveticaneue.ttf
rm gray_letters/helveticaneue-light-001.ttf
#zip -r gray_letters.zip gray_letters
#rm -r gray_letters
