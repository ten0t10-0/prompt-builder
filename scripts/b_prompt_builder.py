import modules.scripts as scripts
import gradio as gr
import os

from abc import ABC, abstractmethod
from modules import scripts
from modules.processing import StableDiffusionProcessing, process_images

def addPrompt(promptExisting: str, promptToAdd: str):
    if len(promptToAdd) > 0:
        if len(promptExisting) > 0:
            promptExisting += ", " + promptToAdd
        else:
            promptExisting = promptToAdd
    
    return promptExisting

def buildRangePrompt(promptA: str, promptB: str, value: float):
    if len(promptA) == 0 or len(promptB) == 0:
        return ""
    
    if value < 0 or value > 1:
        return ""
    
    if value == 0:
        return promptA
    
    if value == 1:
        return promptB
        
    return f"[{promptB}:{promptA}:{value}]"

class B_Prompt(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def getPositive(self, componentMap: dict):
        pass
    
    @abstractmethod
    def getNegative(self, componentMap: dict):
        pass

class B_Prompt_Simple(B_Prompt):
    @staticmethod
    def _fromArgs(**kwargs):
        return B_Prompt_Simple(
            promptPositive = kwargs.get("p", "")
            , promptNegative = kwargs.get("n", "")
        )
    
    @staticmethod
    def _createEmpty():
        return B_Prompt_Simple("", "")
    
    def __init__(self, promptPositive: str="", promptNegative: str=""):
        self.positive = promptPositive
        self.negative = promptNegative
    
    def getPositive(self, componentMap: dict):
        return self.positive
    
    def getNegative(self, componentMap: dict):
        return self.negative

class B_Prompt_Link_Slider(B_Prompt):
    @staticmethod
    def _fromArgs(**kwargs):
        return B_Prompt_Link_Slider(
            linkedKey = kwargs["link_target"]
            , promptA = kwargs["a"]
            , promptB = kwargs["b"]
        )
    
    def __init__(self, linkedKey: str, promptA: str, promptB: str):
        self.linkedKey = linkedKey
        self.promptA = promptA
        self.promptB = promptB
        
        self.isNegativePrompt = False
    
    def buildPrompt(self, componentMap: dict):
        component = componentMap.get(self.linkedKey)
        
        if component is None:
            print(f"B_Prompt_Link_Slider: Invalid key - '{self.linkedKey}'")
            return ""
        
        if type(component) is not B_UI_Component_Slider:
            print(f"B_Prompt_Link_Slider: Linked entry is not a slider - '{self.linkedKey}'")
            return ""
        
        return buildRangePrompt(self.promptA, self.promptB, component.value)
    
    def getPositive(self, componentMap: dict):
        if self.isNegativePrompt:
            return ""
        
        return self.buildPrompt(componentMap)
    
    def getNegative(self, componentMap: dict):
        if not self.isNegativePrompt:
            return ""
        
        return self.buildPrompt(componentMap)

class B_UI(ABC):
    _identifier: int = 0
    
    @staticmethod
    def _buildSeparator():
        return gr.Markdown("<hr style=\"margin: 0.5em 0; border-style: dotted; border-color: var(--border-color-primary);\" />")
    
    def __init__(self, name: str, visible: bool):
        self.identifier = self.getNextIdentifier()
        self.name = self.handleName(name)
        self.visible = visible
    
    def getNextIdentifier(self) -> int:
        B_UI._identifier += 1
        return B_UI._identifier
    
    def handleName(self, name: str = None):
        if name is None:
            name = f"{self.identifier}_{self.getDefaultName()}"
        
        return name
    
    @abstractmethod
    def getDefaultName(self) -> str:
        pass
    
    @abstractmethod
    def buildUI(self):
        pass

class B_UI_Component(B_UI, ABC):
    def __init__(self, name: str = None, defaultValue = None, visible: bool = True):
        self.value = defaultValue
        self.defaultValue = defaultValue
        
        self.component = None
        
        super().__init__(name, visible)
    
    def handleUpdateValue(self, value):
        return value if value is not None else self.defaultValue
    
    def buildUI(self):
        component = self.buildComponent()
        self.component = component[0]
        return component
    
    @abstractmethod
    def getDefaultName(self) -> str:
        pass
    
    @abstractmethod
    def buildComponent(self):
        pass
    
    @abstractmethod
    def setValue(self, value):
        pass
    
    @abstractmethod
    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict):
        pass
    
    @abstractmethod
    def getUpdate(self, value = None):
        pass

class B_UI_Component_Textbox(B_UI_Component):
    @staticmethod
    def _fromArgs(name: str, **kwargs):
        return B_UI_Component_Textbox(
            name = name
            , defaultValue = kwargs.get("v", "")
            , isNegativePrompt = bool(int(kwargs.get("is_negative", 0)))
            , scale = int(kwargs.get("scale", 1))
            , visible = not bool(int(kwargs.get("hide", 0)))
        )
    
    def __init__(self, name: str = None, defaultValue: str = "", isNegativePrompt: bool = False, scale: int = None, visible: bool = True):
        self.isNegativePrompt = isNegativePrompt
        self.scale = scale
        
        super().__init__(name, defaultValue, visible)
    
    def getDefaultName(self) -> str:
        return "Textbox"
    
    def buildComponent(self):
        return [
            gr.Textbox(
                label = self.name
                , value = self.value
                , scale = self.scale
                , visible = self.visible
            )
        ]
    
    def setValue(self, value):
        self.value = value.strip()
    
    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict):
        if not self.isNegativePrompt:
            p.prompt = addPrompt(p.prompt, self.value)
        else:
            p.negative_prompt = addPrompt(p.negative_prompt, self.value)
    
    def getUpdate(self, value = None):
        return gr.Textbox.update(value = self.handleUpdateValue(value))

class B_UI_Component_Dropdown(B_UI_Component):
    @staticmethod
    def _fromArgs(name: str, choicesMap: dict[str, B_Prompt], **kwargs):
        defaultValues = kwargs.get("v", None)
        if defaultValues is not None:
            defaultValues = list(map(lambda v: v.strip(), defaultValues.split(",")))
        
        return B_UI_Component_Dropdown(
            name = name
            , choicesMap = choicesMap
            , defaultValues = defaultValues
            , multiselect = bool(int(kwargs.get("multi_select", 0)))
            , allowCustomValues = bool(int(kwargs.get("allow_custom", 1)))
            , sortChoices = bool(int(kwargs.get("sort", 1)))
            , hideLabel = bool(int(kwargs.get("hide_label", 0)))
            , scale = int(kwargs.get("scale", 1))
            , visible = not bool(int(kwargs.get("hide", 0)))
        )
    
    @staticmethod
    def _buildColorChoicesMap(postfixPrompt: str = ""):
        postfixPrompt = postfixPrompt.strip()
        
        if len(postfixPrompt) > 0:
            postfixPrompt = f" {postfixPrompt}"
        
        return {
            "Dark": B_Prompt_Simple(f"dark{postfixPrompt}")
            , "Light": B_Prompt_Simple(f"light{postfixPrompt}")
            , "Black": B_Prompt_Simple(f"black{postfixPrompt}")
            , "Grey": B_Prompt_Simple(f"grey{postfixPrompt}")
            , "White": B_Prompt_Simple(f"white{postfixPrompt}")
            , "Brown": B_Prompt_Simple(f"brown{postfixPrompt}")
            , "Blue": B_Prompt_Simple(f"blue{postfixPrompt}")
            , "Green": B_Prompt_Simple(f"green{postfixPrompt}")
            , "Red": B_Prompt_Simple(f"red{postfixPrompt}")
            , "Blonde": B_Prompt_Simple(f"blonde{postfixPrompt}")
            , "Rainbow": B_Prompt_Simple(f"rainbow{postfixPrompt}")
            , "Pink": B_Prompt_Simple(f"pink{postfixPrompt}")
            , "Purple": B_Prompt_Simple(f"purple{postfixPrompt}")
            , "Orange": B_Prompt_Simple(f"orange{postfixPrompt}")
            , "Yellow": B_Prompt_Simple(f"yellow{postfixPrompt}")
        }
    
    def __init__(
        self
        , name: str = None
        , choicesMap: dict = {}
        , defaultValues: str | list[str] = None
        , multiselect: bool = False
        , allowCustomValues: bool = True
        , sortChoices: bool = True
        , hideLabel: bool = False
        , scale: int = None
        , visible: bool = True
    ):
        self.choicesMap = self.buildChoicesMap(choicesMap, not multiselect)
        self.multiselect = multiselect
        self.allowCustomValues = allowCustomValues if not multiselect else False
        self.sortChoices = sortChoices
        self.hideLabel = hideLabel
        self.scale = scale
        
        super().__init__(name, self.buildDefaultValue(choicesMap, defaultValues), visible)
    
    def buildChoicesMap(self, choicesMap: dict, insertEmptyChoice: bool):
        choicesMapFinal = choicesMap
        
        if insertEmptyChoice:
            choicesMapFinal = {
                "-": B_Prompt_Simple._createEmpty()
            }
            
            choicesMapFinal.update(choicesMap)
        
        return choicesMapFinal
    
    def buildDefaultValue(self, choicesMap: dict, defaultValues: str | list[str]):
        defaultValue = None if not self.multiselect else []
        
        if (type(defaultValues) is str or defaultValues is None):
            defaultValues = [defaultValues]
        
        if len(choicesMap) > 0:
            choices = list(choicesMap)
            if self.multiselect:
                if defaultValues[0] is not None:
                    defaultValue = []
                    for v in defaultValues:
                        if v in choices:
                            defaultValue.append(v)
            else:
                defaultValueFirst = defaultValues[0]
                if defaultValueFirst is not None and defaultValueFirst in choices:
                    defaultValue = defaultValueFirst
                else:
                    defaultValue = list(self.choicesMap)[0]
        
        return defaultValue
    
    def getChoices(self):
        choices = list(self.choicesMap)
        
        if not self.sortChoices:
            return choices
        
        choicesSorted = []
        
        if choices[0] == "-":
            choicesSorted.append(choices.pop(0))
        
        choices.sort()
        choicesSorted += choices
        
        return choicesSorted
    
    def getBPromptsFromValue(self):
        bPrompts = []
        
        if type(self.value) is str:
            bPrompts.append(self.choicesMap.get(self.value, None if not self.allowCustomValues else B_Prompt_Simple(self.value)))
        elif type(self.value) is list and len(self.value) > 0:
            for k in list(self.choicesMap):
                if k not in self.value:
                    continue
                
                bPrompts.append(self.choicesMap[k])
        
        return bPrompts
    
    def getDefaultName(self) -> str:
        return "Dropdown"
    
    def buildComponent(self):
        return [
            gr.Dropdown(
                label = self.name
                , choices = self.getChoices()
                , multiselect = self.multiselect
                , value = self.value
                , allow_custom_value = self.allowCustomValues
                , show_label = not self.hideLabel
                , scale = self.scale
                , visible = self.visible
            )
        ]
    
    def setValue(self, value):
        self.value = value
    
    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict):
        bPrompts = self.getBPromptsFromValue()
        if len(bPrompts) > 0:
            for bPrompt in bPrompts:
                if bPrompt is not None:
                    p.prompt = addPrompt(p.prompt, bPrompt.getPositive(componentMap))
                    p.negative_prompt = addPrompt(p.negative_prompt, bPrompt.getNegative(componentMap))
    
    def getUpdate(self, value = None):
        return gr.Dropdown.update(value = self.handleUpdateValue(value))

class B_UI_Component_Slider(B_UI_Component):
    @staticmethod
    def _fromArgs(name: str, **kwargs):
        return B_UI_Component_Slider(
            name = name
            , promptA = kwargs.get("a", "")
            , promptB = kwargs.get("b", "")
            , defaultValue = float(kwargs.get("v", -1))
            , isRequired = bool(int(kwargs.get("is_required", 0)))
            , visible = not bool(int(kwargs.get("hide", 0)))
            , buildButtons = bool(int(kwargs.get("build_buttons", 1)))
        )
    
    def __init__(self, name: str = None, promptA: str = "", promptB: str = "", defaultValue: float=None, isRequired: bool=False, visible: bool = True, buildButtons: bool = True):
        self.promptA = promptA.strip()
        self.promptB = promptB.strip()
        self.isRequired = isRequired
        self.buildButtons = buildButtons
        
        self.isNegativePrompt = False
        
        super().__init__(name, defaultValue, visible)
    
    def getMinimum(self):
        return -1 if not self.isRequired else 0
    
    def getMaximum(self):
        return 100
    
    def getStep(self):
        return 1
    
    def buildButton(self, component, text: str, value: float):
        btn = gr.Button(value = text)
        btn.click(
            fn = lambda component: self.getUpdate(value)
            , inputs = component
            , outputs = component
        )
        return btn
    
    def getDefaultName(self) -> str:
        return "Slider"
    
    def buildComponent(self):
        slider = gr.Slider(
            label = self.name
            , minimum = self.getMinimum()
            , maximum = self.getMaximum()
            , value = self.value
            , step = self.getStep()
            , visible = self.visible
        )
        
        with gr.Row():
            self.buildButton(slider, "Male", 0)
            self.buildButton(slider, "Female", self.getMaximum())
        
        return [
            slider
        ]
    
    def setValue(self, value):
        value = float(value)
        self.value = round(value / self.getMaximum(), 2) if value > -1 else value
    
    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict):
        promptToAdd = buildRangePrompt(self.promptA, self.promptB, self.value)
        
        if not self.isNegativePrompt:
            p.prompt = addPrompt(p.prompt, promptToAdd)
        else:
            p.negative_prompt = addPrompt(p.prompt, promptToAdd)
    
    def getUpdate(self, value = None):
        return gr.Slider.update(value = self.handleUpdateValue(value))

class B_UI_Container(B_UI):
    def __init__(self, name: str = None, items: list[B_UI] = [], visible: bool = True, buildResetButton: bool = False, buildCustomPromptInputs: bool = False):
        self.items = self.handleItems(items, buildCustomPromptInputs, name)
        self.buildResetButton = buildResetButton
        
        super().__init__(name, visible)
    
    def handleItems(self, items: list[B_UI], buildCustomPromptInputs: bool, prefix: str):
        if buildCustomPromptInputs:
            items.append(B_UI_Component_Textbox(f"{prefix} - Positive prompt"))
            items.append(B_UI_Component_Textbox(f"{prefix} - Negative prompt", isNegativePrompt = True))
        
        return items
    
    def getBComponents(self) -> list[B_UI_Component]:
        bComponents: list[B_UI_Component] = []
        
        for item in self.items:
            item_type = type(item)
            
            if issubclass(item_type, B_UI_Component):
                bComponents.append(item)
                continue
            
            if issubclass(item_type, B_UI_Container):
                bComponents += item.getBComponents()
                continue
        
        return bComponents
    
    def buildComponents(self):
        components = []
        
        for item in self.items:
            item_ui = item.buildUI()
            
            if len(item_ui) > 0:
                components += item_ui
        
        if self.buildResetButton:
            B_UI._buildSeparator()
            btnReset = self.buildButton_Reset(components)
        
        return components
    
    def resetComponentsValues(self, *args):
        updates = []
        
        bComponents = self.getBComponents()
        for x in bComponents:
            updates.append(x.getUpdate())
        
        if len(updates) == 1:
            updates = updates[0]
        
        return updates
    
    def buildButton_Reset(self, components: list):
        btnReset = gr.Button(value = f"Reset {self.name}")
        
        btnReset.click(
            fn = self.resetComponentsValues
            , inputs = components
            , outputs = components
        )
        
        return btnReset
    
    def buildUI(self):
        components = []
        
        with self.buildContainer():
            components = self.buildComponents()
        
        return components
    
    @abstractmethod
    def getDefaultName(self) -> str:
        pass
    
    @abstractmethod
    def buildContainer(self):
        pass

class B_UI_Container_Tab(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs):
        return B_UI_Container_Tab(
            name = name
            , items = items
            , visible = not bool(int(kwargs.get("hide", 0)))
            , buildResetButton = bool(int(kwargs.get("build_reset_button", 1)))
        )
    
    def __init__(self, name: str = None, items: list[B_UI] = [], visible: bool = True, buildResetButton: bool = True):
        super().__init__(name, items, visible, buildResetButton)
    
    def getDefaultName(self) -> str:
        return "Tab"
    
    def buildContainer(self):
        return gr.Tab(self.name)

class B_UI_Container_Row(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs):
        return B_UI_Container_Row(
            items = items
            , visible = not bool(int(kwargs.get("hide", 0)))
            , buildResetButton = bool(int(kwargs.get("build_reset_button", 0)))
            , name = name
        )
    
    def __init__(self, items: list[B_UI] = [], visible: bool = True, buildResetButton: bool = False, name: str = None):
        super().__init__(name, items, visible, buildResetButton)
    
    def getDefaultName(self) -> str:
        return "Row"
    
    def buildContainer(self):
        return gr.Row(visible = self.visible)

class B_UI_Container_Column(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs):
        return B_UI_Container_Column(
            items = items
            , scale = int(kwargs.get("scale", 1))
            , visible = not bool(int(kwargs.get("hide", 0)))
            , buildResetButton = bool(int(kwargs.get("build_reset_button", 0)))
            , name = name
        )
    
    def __init__(self, items: list[B_UI] = [], scale: int = 1, visible: bool = True, buildResetButton: bool = False, name: str = None):
        self.scale = scale
        
        super().__init__(name, items, visible, buildResetButton)
    
    def getDefaultName(self) -> str:
        return "Column"
    
    def buildContainer(self):
        return gr.Column(scale = self.scale, visible = self.visible)

class B_UI_Container_Group(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs):
        return B_UI_Container_Group(
            items = items
            , visible = not bool(int(kwargs.get("hide", 0)))
            , buildResetButton = bool(int(kwargs.get("build_reset_button", 0)))
            , name = name
        )
    
    def __init__(self, items: list[B_UI] = [], visible: bool = True, buildResetButton: bool = False, name: str = None):
        super().__init__(name, items, visible, buildResetButton)
    
    def getDefaultName(self) -> str:
        return "Group"
    
    def buildContainer(self):
        return gr.Group(visible = self.visible)

class B_UI_Container_Accordion(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs):
        return B_UI_Container_Accordion(
            items = items
            , visible = not bool(int(kwargs.get("hide", 0)))
            , buildResetButton = bool(int(kwargs.get("build_reset_button", 0)))
            , name = name
        )
    
    def __init__(self, name: str = None, items: list[B_UI] = [], visible: bool = True, buildResetButton: bool = False):
        super().__init__(name, items, visible, buildResetButton)
    
    def getDefaultName(self) -> str:
        return "Accordion"
    
    def buildContainer(self):
        return gr.Accordion(label = self.name, visible = self.visible)

class B_UI_Preset():
    @staticmethod
    def _fromArgs(mappings: dict[str, list], **kwargs):
        return B_UI_Preset(
            mappings = mappings
            , isAdditive = bool(int(kwargs.get("is_additive", 0)))
        )
    
    def __init__(self, mappings: dict[str, list], isAdditive: bool = False):
        self.mappings = mappings
        self.isAdditive = isAdditive

class B_UI_Builder(ABC):
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.args = kwargs
    
    @abstractmethod
    def build(self) -> B_UI | B_UI_Preset:
        pass

class B_UI_Builder_WithParent(B_UI_Builder):
    def __init__(self, name: str, parent: B_UI_Builder, **kwargs):
        self.parent = parent
        
        super().__init__(name, **kwargs)
    
    @abstractmethod
    def build(self) -> B_UI:
        pass

class B_UI_Container_Builder(B_UI_Builder_WithParent):
    def __init__(self, t: type[B_UI_Container], name: str, parent: B_UI_Builder, **kwargs):
        self.t = t
        
        self.builtChildren: list[B_UI] = []
        
        super().__init__(name, parent, **kwargs)
    
    def finalizeBuilt(self, built: B_UI):
        if self.parent is not None:
            self.parent.builtChildren.append(built)
        
        return built
    
    def build(self) -> B_UI:
        return self.finalizeBuilt(self.t._fromArgs(self.name, self.builtChildren, **self.args))

class B_UI_Component_Builder(B_UI_Builder_WithParent):
    def __init__(self, t: type[B_UI_Component], name: str, parent: B_UI_Container_Builder, **kwargs):
        self.t = t
        
        super().__init__(name, parent, **kwargs)
    
    def build(self) -> B_UI:
        builtSelf = self.t._fromArgs(self.name, **self.args)
        
        if self.parent is not None:
            self.parent.builtChildren.append(builtSelf)
        
        return builtSelf

class B_UI_Component_Dropdown_Builder(B_UI_Builder_WithParent):
    def __init__(self, name: str, parent: B_UI_Container_Builder, **kwargs):
        self.choicesMap = {}
        
        super().__init__(name, parent, **kwargs)
    
    def addChoice(self, text: str, **kwargs):
        bPrompt: B_Prompt = None
        
        link_type = kwargs.get("link_type", None)
        match link_type:
            case "SLIDER":
                bPrompt = B_Prompt_Link_Slider._fromArgs(**kwargs)
            case _:
                bPrompt = B_Prompt_Simple._fromArgs(**kwargs)
        
        self.choicesMap[text] = bPrompt
    
    def addChoices(self, **kwargs):
        choicesMap = {}
        
        special_type = kwargs["type"]
        match special_type:
            case "COLOR":
                choicesMap = B_UI_Component_Dropdown._buildColorChoicesMap(kwargs["postfix"])
            case _:
                print(f"Invalid CHOICES type: {special_type}")
        
        self.choicesMap = choicesMap
    
    def build(self) -> B_UI:
        builtSelf = B_UI_Component_Dropdown._fromArgs(self.name, self.choicesMap, **self.args)
        
        if self.parent is not None:
            self.parent.builtChildren.append(builtSelf)
        
        return builtSelf

class B_UI_Preset_Builder(B_UI_Builder):
    def __init__(self, name: str, **kwargs):
        self.mappings = {}
        
        super().__init__(name, **kwargs)
    
    def addMapping(self, name: str, **kwargs):
        value = kwargs.get("v", [])
        
        if len(value) > 0:
            value = list(map(lambda v: v.strip(), value.split(",")))
        
        self.mappings[name] = value
    
    def build(self) -> B_UI_Preset:
        return B_UI_Preset._fromArgs(self.mappings, **self.args)

class B_UI_Map():
    def __init__(self, path_base: str, file_name_layout: str, file_name_presets: str, tagged_show: bool = True):
        self.layout = self.parseLayout(os.path.join(path_base, file_name_layout), tagged_show)
        self.presets = self.parsePresets(os.path.join(path_base, file_name_presets))
        
        self.componentMap = {}
        
        self.buildComponentMapRecursive(self.layout, self.componentMap)
    
    def readLine(self, l: str) -> tuple[str, str, dict[str, str]]:
        l = l.strip()
        
        l_type = l
        l_name = None
        l_args = {}
        
        if len(l) > 0:
            index = l.find(" ")
            if index != -1:
                l_type = l[:index]
            
            l = l[len(l_type) + 1:]
            
            l_arg_index = l.find("--")
            if l_arg_index == -1:
                l_name = l
            elif l_arg_index > 0:
                l_name = l[:l_arg_index - 1 if l_arg_index > -1 else len(l)]
                l = l[len(l_name) + 1:]
            
            l_args = {}
            for l_arg in l.split("--")[1:]:
                l_arg_name = l_arg[:l_arg.index(" ")]
                l_arg_value = l_arg[len(l_arg_name) + 1:].strip()
                l_args[l_arg_name] = l_arg_value
        
        return l_type, l_name, l_args
    
    def parseLayout(self, file_path_layout: str, tagged_show: bool) -> list[B_UI]:
        layout: list[B_UI] = []
        
        builder_current_container: B_UI_Container_Builder = None
        builder_current_dropdown: B_UI_Component_Dropdown_Builder = None
        
        skip = 0
        
        def _build(builder: B_UI_Builder_WithParent):
            built = builder.build()
            if builder.parent is None:
                layout.append(built)
            return built
        
        with open(file_path_layout) as file_layout:
            for l in file_layout:
                l_type, l_name, l_args = self.readLine(l)
                
                if len(l_type) == 0:
                    continue
                    
                if l_type == ".":
                    break
                
                if l_type == "END":
                    if skip == 0 and builder_current_dropdown is not None:
                        built = _build(builder_current_dropdown)
                        builder_current_dropdown = None
                        continue
                    
                    if skip == 0 and builder_current_container is not None:
                        built = _build(builder_current_container)
                        builder_current_container = builder_current_container.parent
                        continue
                    
                    skip -= 1
                    
                    continue
                
                if l_args.get("x", None) == "1":
                    if not tagged_show:
                        skip += 1
                        continue
                
                match l_type:
                    case "TEXTBOX":
                        built = _build(B_UI_Component_Builder(B_UI_Component_Textbox, l_name, builder_current_container, **l_args))
                    
                    case "SLIDER":
                        built = _build(B_UI_Component_Builder(B_UI_Component_Slider, l_name, builder_current_container, **l_args))
                    
                    case "DROPDOWN":
                        if skip > 0:
                            skip += 1
                            continue
                        
                        builder_current_dropdown = B_UI_Component_Dropdown_Builder(l_name, builder_current_container, **l_args)
                    
                    case "CHOICE":
                        if skip > 0:
                            continue
                        
                        builder_current_dropdown.addChoice(l_name, **l_args)
                    
                    case "CHOICES":
                        if skip > 0:
                            continue
                        
                        builder_current_dropdown.addChoices(**l_args)
                    
                    case "GROUP":
                        if skip > 0:
                            skip += 1
                            continue
                        
                        builder_current_container = B_UI_Container_Builder(B_UI_Container_Group, l_name, builder_current_container, **l_args)
                    
                    case "TAB":
                        if skip > 0:
                            skip += 1
                            continue
                        
                        builder_current_container = B_UI_Container_Builder(B_UI_Container_Tab, l_name, builder_current_container, **l_args)
                    
                    case "ROW":
                        if skip > 0:
                            skip += 1
                            continue
                        
                        builder_current_container = B_UI_Container_Builder(B_UI_Container_Row, l_name, builder_current_container, **l_args)
                    
                    case "COLUMN":
                        if skip > 0:
                            skip += 1
                            continue
                        
                        builder_current_container = B_UI_Container_Builder(B_UI_Container_Column, l_name, builder_current_container, **l_args)
                    
                    case "ACCORDION":
                        if skip > 0:
                            skip += 1
                            continue
                        
                        builder_current_container = B_UI_Container_Builder(B_UI_Container_Accordion, l_name, builder_current_container, **l_args)
                    
                    case _:
                        print(f"Invalid layout type: {l_type}")
        
        return layout
    
    def parsePresets(self, file_path_presets: str) -> dict[str, B_UI_Preset]:
        presets: dict[str, B_UI_Preset] = {}
        
        builder_current: B_UI_Preset_Builder = None
        
        with open(file_path_presets) as file_presets:
            for l in file_presets:
                l_type, l_name, l_args = self.readLine(l)
                
                if len(l_type) == 0:
                    continue
                    
                if l_type == ".":
                    break
                
                if l_type == "END":
                    presets[builder_current.name] = builder_current.build()
                    builder_current = None
                    continue
                
                match l_type:
                    case "PRESET":
                        builder_current = B_UI_Preset_Builder(l_name, **l_args)
                    
                    case "SET":
                        builder_current.addMapping(l_name, **l_args)
                    
                    case _:
                        print(f"Invalid preset type: {l_type}")
        
        return presets
    
    def buildComponentMapRecursive(self, source: list[B_UI], target: dict[str, B_UI_Component]):
        for x in source:
            x_type = type(x)
            
            x_isComponent = issubclass(x_type, B_UI_Component)
            x_isContainer = issubclass(x_type, B_UI_Container)
            
            x_isLabeled = (
                x_isComponent
                or issubclass(x_type, B_UI_Container_Tab)
                or issubclass(x_type, B_UI_Container_Accordion)
            )
            
            key = ""
            if x_isLabeled:
                key = x.name
            
            if x_isComponent:
                if key in target:
                    print(f"WARNING: Duplicate attribute -> {key}")
                
                target[key] = x
                
                continue
            
            if x_isContainer:
                self.buildComponentMapRecursive(x.items, target)
                continue
    
    def buildUI(self):
        B_UI._buildSeparator()
        
        components = []
        
        for item in self.layout:
            item_ui = item.buildUI()
            
            if len(item_ui) > 0:
                components += item_ui
        
        self.buildPresetsUI(components)
        
        return components
    
    def buildPresetsUI(self, components: list):
        presetKeys = self.presets.keys()
        
        if len(presetKeys) == 0:
            return
        
        def applyPreset(presetKey: str, *args):
            preset = self.presets[presetKey]
            
            def getValue(component, componentValue):
                hasPresetValue = component.label in preset.mappings
                if not hasPresetValue:
                    return self.componentMap[component.label].defaultValue if not preset.isAdditive else componentValue
                
                presetValue = preset.mappings[component.label]
                
                if type(component) is not gr.Dropdown or not component.multiselect:
                    presetValue = presetValue[0]
                
                return presetValue
            
            return list(map(getValue, components, args))
        
        B_UI._buildSeparator()
        with gr.Accordion("Presets", open = True):
            i = 0
            for presetKey in presetKeys:
                button_preset = gr.Button(presetKey)
                button_preset.click(
                    fn = applyPreset
                    , inputs = [button_preset] + components
                    , outputs = components
                )
                
                i += 1
                if i < len(presetKeys):
                    B_UI._buildSeparator()

b_layout = B_UI_Map(
    path_base = os.path.join(scripts.basedir(), "scripts", "b_prompt_builder")
    , file_name_layout = "layout.txt"
    , file_name_presets = "presets.txt"
    , tagged_show = True
)

class Script(scripts.Script):
    def title(self):
        return "B Prompt Builder"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        return b_layout.buildUI()

    def run(self, p, *args):
        components = []
        
        i = 0
        for k in list(b_layout.componentMap):
            component = b_layout.componentMap[k]
            component.setValue(args[i])
            component.handlePrompt(p, b_layout.componentMap)
            i = i+1
        
        proc = process_images(p)
        
        return proc
