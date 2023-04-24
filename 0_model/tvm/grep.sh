egrep -A2 "Execution time summary|model_path" ./out.txt | egrep "model_path|   " | grep -v "mean" # | grep  "model"
echo 
egrep -A2 "Execution time summary|model_path" ./out.txt | egrep "model_path|   " | grep -v "mean" | grep -v "model"
echo 
egrep -A2 "Execution time summary|model_path" ./out.txt | egrep "model_path|   " | grep -v "mean" | grep "model" | tr '/' ' ' | awk '{print $6}' | tr '.' ' ' | column -t 
