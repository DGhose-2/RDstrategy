The file \texttt{main.ipynb} can be used for data preparation, cleaning, feature-engineering, and generating the final models.

Ensure that utility files
\begin{itemize}
    \item \texttt{data\_utils.py}
    \item \texttt{nlp\_utils.py}
    \item \texttt{model\_utils.py}
\end{itemize}
are all in the same working directory as \texttt{main.ipynb}. File \texttt{requirements.txt} lists packages and versions required for reproducibility.

\medskip
\texttt{GoogleScholar.ipynb} contains the code for extracting project outcome information from Google Scholar, which depends upon library \texttt{scholarly}.
\texttt{scholarly} searches are often obstructed by Google's reCAPTCHA, resulting in IP bans, especially if searching by title rather than by author.
