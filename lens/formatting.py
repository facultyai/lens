"""Table formatting for Jupyter notebooks"""

# Copyright (c) 2012-2013, Eric Moyer <eric@lemoncrab.com>
# Copyright (c) 2016, ASI Data Science
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# Neither the name of the ipy_table Development Team nor the names of
# its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys


class JupyterTable(object):

    _valid_borders = {"left", "right", "top", "bottom", "all"}

    def __init__(self, array, caption=""):
        self.array = array
        self.caption = caption

        self._num_rows = len(array)
        self._num_columns = len(array[0])

        # Check that array is well formed
        for row in array:
            if len(row) != self._num_columns:
                raise ValueError("Array rows must all be of equal length.")

        self._cell_styles = [
            [{"float_format": "%0.4f"} for dummy in range(self._num_columns)]
            for dummy2 in range(self._num_rows)
        ]

    def _repr_html_(self):
        """Jupyter display protocol: HTML representation.

        The Jupyter display protocol calls this method to get the HTML
        representation of this object.
        """
        # Generate TABLE tag (<tr>)
        html = (
            self.caption
            + '<table border="1" cellpadding="3" cellspacing="0" '
            + ' style="border:1px solid black;border-collapse:collapse;">'
        )

        for row, row_data in enumerate(self.array):

            # Generate ROW tag (<tr>)
            html += "<tr>"
            for (column, item) in enumerate(row_data):
                if not _key_is_valid(
                    self._cell_styles[row][column], "suppress"
                ):

                    # Generate CELL tag (<td>)
                    # Apply floating point formatter to the cell contents
                    # (if it is a float)
                    item_html = self._formatter(
                        item, self._cell_styles[row][column]
                    )

                    # Add bold and italic tags if set
                    if _key_is_valid(self._cell_styles[row][column], "bold"):
                        item_html = "<b>" + item_html + "</b>"
                    if _key_is_valid(self._cell_styles[row][column], "italic"):
                        item_html = "<i>" + item_html + "</i>"

                    # Get html style string
                    style_html = self._get_style_html(
                        self._cell_styles[row][column]
                    )

                    # Append cell
                    html += "<td" + style_html + ">" + item_html + "</td>"
            html += "</tr>"
        html += "</table>"
        return html

    def _get_style_html(self, style_dict):
        """Parse the style dictionary and return equivalent html style text."""
        style_html = ""
        if _key_is_valid(style_dict, "color"):
            style_html += "background-color:" + style_dict["color"] + ";"

        if _key_is_valid(style_dict, "thick_border"):
            for edge in self._split_by_comma(style_dict["thick_border"]):
                style_html += "border-%s: 3px solid black;" % edge

        if _key_is_valid(style_dict, "no_border"):
            for edge in self._split_by_comma(style_dict["no_border"]):
                style_html += "border-%s: 1px solid transparent;" % edge

        if _key_is_valid(style_dict, "align"):
            style_html += "text-align:" + str(style_dict["align"]) + ";"

        if _key_is_valid(style_dict, "width"):
            style_html += "width:" + str(style_dict["width"]) + "px;"

        if style_html:
            style_html = ' style="' + style_html + '"'

        if _key_is_valid(style_dict, "row_span"):
            style_html = (
                'rowspan="' + str(style_dict["row_span"]) + '";' + style_html
            )

        if _key_is_valid(style_dict, "column_span"):
            style_html = (
                'colspan="'
                + str(style_dict["column_span"])
                + '";'
                + style_html
            )

        # Prepend a space if non-blank
        if style_html:
            return " " + style_html
        return ""

    def _formatter(self, item, cell_style):
        """Apply formatting to cell contents.

        Applies float format to item if item is a float or float64.
        Converts spaces to non-breaking if wrap is not enabled.
        Returns string.
        """

        # The following check is performed as a string comparison
        # so that ipy_table does not need to require (import) numpy.
        if (
            str(type(item)) in ["<type 'float'>", "<type 'numpy.float64'>"]
            and "float_format" in cell_style
        ):
            text = cell_style["float_format"] % item
        else:
            if isinstance(item, str):
                text = item
            else:
                text = str(item)

        if sys.version_info.major < 3:
            # QA disabled as unicode is a NameError in Python 3.
            text = unicode(text, encoding="utf-8")  # noqa

        # If cell wrapping is not specified
        if not ("wrap" in cell_style and cell_style["wrap"]):
            # Convert all spaces to non-breaking and return
            text = text.replace(" ", "&nbsp")
        return text

    def _split_by_comma(self, comma_delimited_text):
        """Returns a list of the words in the comma delimited text."""
        return comma_delimited_text.replace(" ", "").split(",")


def _key_is_valid(dictionary, key):
    """Test that a dictionary key exists and that its value is not blank."""
    if key in dictionary:
        if dictionary[key]:
            return True
    return False
