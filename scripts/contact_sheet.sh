cd ../stats
FILES=$(ls)
for i in $FILES
do
    
   if [ -e "$i/true_negatives-$i-test.pdf" ]
   then
     echo "$i found"
   else
       if [ -e "$i/$i-test_results.csv" -a -e "$i/$i-train_results.csv" ]
       then
           python ../py_scripts/7_contact_sheet.py -d $i/$i-test_results.csv -o $i-test -f $i -s 100
           python ../py_scripts/7_contact_sheet.py -d $i/$i-train_results.csv -o $i-train -f $i -s 100
       else
           echo "$i No csv files"
       fi
   fi
done
