 Files excluded due to corruption in dataset:
 	098565.mp3
 	098567.mp3
 	098569.mp3
 	099139.mp3
 	107335.mp3
 	108925.mp3
 	133297.mp3
 	
Data files are encoded as follows:

{
    [songNumber] : {
        'data' : [2D np.array containing normalized data],
        'genre' : [np.array containing One hot encoded genre classification]
    }
}

The order of genres is: ["International", "Pop", "Rock", "Electronic", "Folk", "Hip-Hop", "Experimental", "Instrumental"]

 To train network:
 	- Extract FMA_small dataset to Songs/ directory within the Data/ directory, and move all files from their number folders into the Songs/ directory
 	- Run getFiles.py and wait for the preprocessing to complete, progress will be reported along with ETA
 	- Run trainModel.py, optionally, run tnesorboard as well to monitor training

 To validate network:
 	- Using a single file:
 		- Run python getClassification.py [pathToMP3].mp3
 		- Wait for classification to be returned in english

 	- Using a set of songs
 		- Run python getClassification.py [pathToPreprocessedFile].pkl
 		- Wait for validation accuracy to be computed
 		- (Note, you can use the generated files here, just download them as mentioned in the report and place them within the Data/ directory)
