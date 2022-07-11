# K2 & OAI

# FAQ

* **I get errors when I import modules, likely due to the code being stored in `src`.**

If you are using PyCharm, make sure to check the `src` folder as Source root, as explained [here](https://www.jetbrains.com/help/pycharm/configuring-project-structure.html) and [here](https://www.jetbrains.com/help/pycharm/content-root.html#root_types=). With VS Code, make sure you set up the ExtraPaths in the settings: see [this](https://stackoverflow.com/a/60892657/12445701) answers on StackOverflow and [here](https://code.visualstudio.com/docs/python/settings-reference) for the official list of settings for Python.

* **How do I launch the dashboard?**

If you are using poetry to manage the dependencies, run:

`poetry run streamlit run main.py`

If you used the `requirements.txt` to instantiate a virtual environment, then it suffices to execute this:

`streamlit run app.py`
