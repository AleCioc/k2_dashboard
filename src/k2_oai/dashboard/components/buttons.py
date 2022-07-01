"""
Dashboard buttons, e.g. to change photos and save data
"""

import numpy as np
import streamlit as st

__all__ = ["choose_roof_id"]


def _load_random_photo(roofs_list):
    st.session_state["roof_id"] = np.random.choice(roofs_list)


def _change_roof_id(how: str, roofs_list):
    # note: np.where returns a tuple, in this case ([array],).
    # use the double indexing like [0][0]!
    current_index: int = np.where(roofs_list == st.session_state["roof_id"])[0][0]

    if how == "next":
        # otherwise is out of index
        if current_index < len(roofs_list) - 1:
            st.session_state["roof_id"] = roofs_list[current_index + 1]
    elif how == "previous":
        if current_index > 0:
            st.session_state["roof_id"] = roofs_list[current_index - 1]
    else:
        raise ValueError(f"Invalid `how`: {how}. Must be `next` or `previous`.")


def choose_roof_id(metadata, remaining_roofs):

    roofs_list = metadata.roof_id.unique()

    st.markdown(
        f"""
        ## :house: Roof identifier

        There are `{roofs_list.shape[0]}` roofs available with these settings.

        Choose a roof id manually...
        """
    )

    chosen_roof_id = st.selectbox(
        "Roof identifier:",
        options=roofs_list,
        help="Choose one out of all the available roof ids",
        key="roof_id",
    )

    st.write("...or use the buttons below.")

    buf, st_previous, st_rand, st_next, buf = st.columns((0.5, 1, 1, 1, 0.5))

    st_previous.button(
        "⬅️",
        help="Load the photo before this one. "
        "If nothing happens, this is the first photo.",
        on_click=_change_roof_id,
        args=("previous", roofs_list),
    )

    st_rand.button(
        "🔀",
        help="Load a random photo that was not labelled yet",
        on_click=_load_random_photo,
        args=(remaining_roofs,),
    )

    st_next.button(
        "➡️",
        help="Load the photo right after this one. "
        "If nothing happens, this is the last photo.",
        on_click=_change_roof_id,
        args=("next", roofs_list),
    )

    return chosen_roof_id
