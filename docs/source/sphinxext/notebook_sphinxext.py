"""This is a modified version of ``notebook_sphinxext.py``

The original was from: https://github.com/ngoldbaum/RunNotebook as of commit
944f983

This is the license for RunNotebook:

Copyright (c) 2013 Nathan Goldbaum. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import errno
import nbformat
import os
import shutil
import tempfile
import uuid
from sphinx.util.compat import Directive
from docutils import nodes
from docutils.parsers.rst import directives
from traitlets.config import Config
from nbconvert import html, python


class NotebookDirective(Directive):
    """Insert an evaluated notebook into a document

    This uses runipy and nbconvert to transform a path to an unevaluated notebook
    into html suitable for embedding in a Sphinx document.
    """
    required_arguments = 1
    optional_arguments = 1
    option_spec = {'skip_exceptions': directives.flag}
    final_argument_whitespace = True

    def run(self):  # check if there are spaces in the notebook name
        nb_path = self.arguments[0]
        if ' ' in nb_path:
            raise ValueError(
                "Due to issues with docutils stripping spaces from links, white "
                "space is not allowed in notebook filenames '{0}'".format(
                    nb_path))
        # check if raw html is supported
        if not self.state.document.settings.raw_enabled:
            raise self.warning('"%s" directive disabled.' % self.name)

        cwd = os.getcwd()
        tmpdir = tempfile.mkdtemp()
        os.chdir(tmpdir)

        # get path to notebook
        nb_filename = self.arguments[0]
        nb_basename = os.path.basename(nb_filename)
        rst_file = self.state_machine.document.attributes['source']
        rst_dir = os.path.abspath(os.path.dirname(rst_file))
        nb_abs_path = os.path.abspath(os.path.join(rst_dir, nb_filename))

        # Move files around.
        rel_dir = os.path.relpath(rst_dir, setup.confdir)
        dest_dir = os.path.join(setup.app.builder.outdir, rel_dir)
        dest_path = os.path.join(dest_dir, nb_basename)

        image_dir, image_rel_dir = make_image_dir(setup, rst_dir)

        # Ensure desination build directory exists
        thread_safe_mkdir(os.path.dirname(dest_path))

        # Copy unevaluated notebook
        shutil.copyfile(nb_abs_path, dest_path)

        # Construct paths to versions getting copied over
        dest_path_eval = dest_path.replace('.ipynb', '_evaluated.ipynb')
        dest_path_script = dest_path.replace('.ipynb', '.py')
        rel_path_eval = nb_basename.replace('.ipynb', '_evaluated.ipynb')
        rel_path_script = nb_basename.replace('.ipynb', '.py')

        # Create python script vesion
        script_text = nb_to_python(nb_abs_path)
        f = open(dest_path_script, 'wb')
        f.write(script_text.encode('utf8'))
        f.close()

        skip_exceptions = 'skip_exceptions' in self.options

        evaluated_text, resources = evaluate_notebook(
            nb_abs_path, dest_path_eval, skip_exceptions=skip_exceptions)

        evaluated_text = write_notebook_output(
            resources, image_dir, image_rel_dir, evaluated_text)

        # Create link to notebook and script files
        link_rst = "(" + \
                   formatted_link(nb_basename) + "; " + \
                   formatted_link(rel_path_eval) + "; " + \
                   formatted_link(rel_path_script) + \
                   ")"

        self.state_machine.insert_input([link_rst], rst_file)

        # create notebook node
        attributes = {'format': 'html', 'source': 'nb_path'}
        nb_node = notebook_node('', evaluated_text, **attributes)
        (nb_node.source, nb_node.line) = \
            self.state_machine.get_source_and_line(self.lineno)

        # add dependency
        self.state.document.settings.record_dependencies.add(nb_abs_path)

        # clean up
        os.chdir(cwd)
        shutil.rmtree(tmpdir, True)

        return [nb_node]


class notebook_node(nodes.raw):
    pass


def nb_to_python(nb_path):
    """convert notebook to python script"""
    exporter = python.PythonExporter()
    output, resources = exporter.from_filename(nb_path)
    return output


def nb_to_html(nb_path, skip_exceptions):
    """convert notebook to html"""

    nbconvert_config = Config({
        'ExtractOutputPreprocessor': {'enabled': True},
        'ExecutePreprocessor': {
            'enabled': True,
            # make this configurable?
            'timeout': 3600,
        }
    })

    if skip_exceptions is False:
        nbconvert_config['ExecutePreprocessor']['allow_errors'] = True

    exporter = html.HTMLExporter(template_file='full', config=nbconvert_config)
    notebook = nbformat.read(nb_path, nbformat.NO_CONVERT)
    output, resources = exporter.from_notebook_node(notebook)
    header = output.split('<head>', 1)[1].split('</head>', 1)[0]
    body = output.split('<body>', 1)[1].split('</body>', 1)[0]

    # http://imgur.com/eR9bMRH
    header = header.replace('<style', '<style scoped="scoped"')
    header = header.replace(
        'body {\n  overflow: visible;\n  padding: 8px;\n}\n', '')
    header = header.replace(
        'div#notebook {\n  overflow: visible;',
        '.container {\n  width: 100%;\n}\ndiv#notebook {\n  overflow: visible;\n  width: 100%;')
    header = header.replace(
        '#notebook-container {\n    padding: 15px;',
        '#notebook-container {\n    padding: 0px;',)
    header = header.replace("code,pre{", "code{")

    # Filter out styles that conflict with the sphinx theme.
    filter_strings = [
        'navbar',
        'body{',
        'alert{',
        'uneditable-input{',
        'collapse{',
    ]

    #filter_strings.extend(['h%s{' % (i+1) for i in range(6)])

    line_begin = [
        'pre{',
        'p{margin'
    ]

    filterfunc = lambda x: not any([s in x for s in filter_strings])
    header_lines = filter(filterfunc, header.split('\n'))

    filterfunc = lambda x: not any([x.startswith(s) for s in line_begin])
    header_lines = filter(filterfunc, header_lines)

    header = '\n'.join(header_lines)

    # concatenate raw html lines
    lines = ['<div class="ipynotebook">']
    lines.append(header)
    lines.append(body)
    lines.append('</div>')
    return '\n'.join(lines), resources, notebook


def evaluate_notebook(nb_path, dest_path, skip_exceptions=True):
    # Create evaluated version and save it to the dest path.
    lines, resources, notebook = nb_to_html(nb_path, skip_exceptions)
    nbformat.write(notebook, dest_path)
    return lines, resources


def formatted_link(path):
    return "`%s <%s>`__" % (os.path.basename(path), path)


def visit_notebook_node(self, node):
    self.visit_raw(node)


def depart_notebook_node(self, node):
    self.depart_raw(node)


def setup(app):
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir

    app.add_node(notebook_node,
                 html=(visit_notebook_node, depart_notebook_node))

    app.add_directive('notebook', NotebookDirective)

    retdict = dict(
        version='0.1',
        parallel_read_safe=True,
        parallel_write_safe=True
    )

    return retdict


def make_image_dir(setup, rst_dir):
    image_dir = setup.app.builder.outdir + os.path.sep + '_images'
    rel_dir = os.path.relpath(setup.confdir, rst_dir)
    image_rel_dir = rel_dir + os.path.sep + '_images'
    thread_safe_mkdir(image_dir)
    return image_dir, image_rel_dir


def write_notebook_output(resources, image_dir, image_rel_dir, evaluated_text):
    my_uuid = uuid.uuid4().hex

    for output in resources['outputs']:
        new_name = image_dir + os.path.sep + my_uuid + output
        new_relative_name = image_rel_dir + os.path.sep + my_uuid + output
        evaluated_text = evaluated_text.replace(output, new_relative_name)
        with open(new_name, 'wb') as f:
            f.write(resources['outputs'][output])
    return evaluated_text


def thread_safe_mkdir(dirname):
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        pass
