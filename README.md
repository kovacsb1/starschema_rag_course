# starschema_rag_course
Repository for HCL-Starschema RAG course. We will code in Python during the course, so make sure you have Python installed on your machine.

## Virtual environment
It is recommended to create a separate virtual environment for the course, so the Python packages won't create conflicts with the packeges use for daily work.
Here is a script to do that:

```
python -m venv .venv
source .venv/bin/activate
```
The second line also activates the virtual environment.

Make sure pip is installed! If the first line fails, run the second one!
```
python -m pip --version
python -m pip install --upgrade pip
```

## `requirements.txt`

The Python packages required are listed in `requirements.txt`. To install it, run the following command:
```
python -m pip install -r requirements.txt
```

## Deactivating the virtual environment
Once you are not working on the course, you can ddeactivate your virtaul environment. Simply run:
```
deactivate
```
