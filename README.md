# kNN Algorithm
## Objective
This is a kNN (k Nearest Neighbours) algorithm which finds k nearest neighbours
to a data entry in a database.

## Libraries Used
 - numpy
 - collections

## Example
Includes an implementation to recommend other movies based on a movie entry's IMBD rating and genre.

For example, using the file `movies.txt` as the database,
for a movie with a rating of 7.2 and falling into the genres 'Biography', 'Drama'
and 'History' the algorithm recommends, out of 30 movies, the following ones:
- A Beautiful Mind
- The Wind Rises
- Hacksaw Ridge
- 12 Years a Slave
- Queen of Katwe

For this example, the input `[7.2, 1, 1, 0, 0, 0, 0, 1, 0]` was passed into the
algorithm `kNN_algorithm`.
