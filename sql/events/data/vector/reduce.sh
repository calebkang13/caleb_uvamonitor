mysql -u monitor_db -D monitor_db -p'monitor_db' -e 'SELECT * FROM hcdm_vectors;' | sed 's/\t/,/g' > hcdm_vectors_reduced.csv;
