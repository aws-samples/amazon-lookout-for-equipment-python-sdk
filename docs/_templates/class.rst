{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: Attributes

    .. autosummary::
    {% for item in attributes %}
       ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block methods %}
    {% if methods %}
    .. rubric:: Methods

    .. autosummary::
    {% for item in methods %}
       ~{{ name }}.{{ item }}
    {%- endfor %}
    
    {% for item in methods %}
    .. automethod:: {{ item }}
    {%- endfor %}
    
    {% endif %}
    {% endblock %}

.. 
    include:: {{module}}.{{objname}}.examples

.. raw:: html

    <div class="clearer"></div>
