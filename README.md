This benchmark generates a large $175000$ by $175000$ sparse matrix where approximately 1% of the values on each row are non zero values uniformly randomly generated between 0 and 1. This benchmark uses the intel Math Kernel Library.
```
make matrix
./matrix <dimension> <num threads>
```
