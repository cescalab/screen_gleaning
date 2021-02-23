#!/bin/bash
mkdir gray_letters
cp helveticaneue.ttf gray_letters
cp helveticaneue-light-001.ttf gray_letters

graytone=0.2
scriptwidth="375pt"
scriptheight="667pt"
dpidensity=326

for Y in {1..2}
do
  #Create random sequence
  rseq=""
  for i in {1..6}
  do
    rseq="$rseq$(($RANDOM%10))-"
  done
  #format the sequence as r-a-n-d-o-m
  rseq=${rseq::-1}
  rseqjoined=$(echo "$rseq" | sed 's/-//g')
  #replace the random sequence in tex file
  sed -e "s/graytone/$graytone/g; s/scriptwidth/$scriptwidth/; s/scriptheight/$scriptheight/; s/herecomestherandom/$rseqjoined/g" xetexMakeSMS.tex > gray_letters/$rseq\_.tex
  cd gray_letters
  xelatex -synctex=1 -interaction=nonstopmode $rseq\_.tex
  convert -antialias -density $dpidensity -quality 100 $rseq\_.pdf $rseq\_.jpg 
  cd ..
  rm gray_letters/$rseq\_.tex
  rm gray_letters/$rseq\_.log
  rm gray_letters/$rseq\_.aux
  rm gray_letters/$rseq\_.synctex.gz
done

rm gray_letters/helveticaneue.ttf
rm gray_letters/helveticaneue-light-001.ttf
#zip -r gray_letters.zip gray_letters
#rm -r gray_letters
