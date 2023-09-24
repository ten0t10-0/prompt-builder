import modules.scripts as scripts
import gradio as gr
import os

from abc import ABC, abstractmethod
from modules import scripts
from modules.processing import StableDiffusionProcessing, process_images

def addPrompt(promptExisting: str, promptToAdd: str) -> str:
    if len(promptToAdd) > 0:
        if len(promptExisting) > 0:
            promptExisting += ", " + promptToAdd
        else:
            promptExisting = promptToAdd
    
    return promptExisting

def buildRangePrompt(promptA: str, promptB: str, value: float) -> str:
    if len(promptA) == 0 or len(promptB) == 0:
        return ""
    
    if value < 0 or value > 1:
        return ""
    
    if value == 0:
        return promptA
    
    if value == 1:
        return promptB
        
    return f"[{promptB}:{promptA}:{value}]"

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
    
    def handleName(self, name: str = None) -> str:
        if name is None:
            name = f"{self.identifier}_{self.getDefaultName()}"
        
        return name
    
    @abstractmethod
    def getDefaultName(self) -> str:
        pass
    
    @abstractmethod
    def buildUI(self) -> any:
        pass

class B_UI_Component(B_UI, ABC):
    @staticmethod
    @abstractmethod
    def _fromArgs(name: str, *args, **kwargs: str):
        pass
    
    def __init__(self, name: str = None, defaultValue = None, visible: bool = True):
        self.value = defaultValue
        self.defaultValue = defaultValue
        
        self.component = None
        
        super().__init__(name, visible)
    
    def handleUpdateValue(self, value):
        return value if value is not None else self.defaultValue
    
    def buildUI(self) -> any:
        component = self.buildComponent()
        self.component = component[0]
        return component
    
    @abstractmethod
    def getDefaultName(self) -> str:
        pass
    
    @abstractmethod
    def buildComponent(self) -> list[any]:
        pass
    
    @abstractmethod
    def setValue(self, value):
        pass
    
    @abstractmethod
    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict):
        pass
    
    @abstractmethod
    def getUpdate(self, value = None) -> any:
        pass

    @abstractmethod
    def finalizeComponent(self, componentMap: dict):
        pass

class B_UI_Container(B_UI):
    @staticmethod
    @abstractmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str):
        pass

    def __init__(self, name: str = None, items: list[B_UI] = [], visible: bool = True, buildResetButton: bool = False, buildCustomPromptInputs: bool = False):
        self.items = self.handleItems(items, buildCustomPromptInputs, name)
        self.buildResetButton = buildResetButton
        
        super().__init__(name, visible)
    
    def handleItems(self, items: list[B_UI], buildCustomPromptInputs: bool, prefix: str) -> list[B_UI]:
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
    
    def buildComponents(self) -> list[any]:
        components = []
        
        for item in self.items:
            item_ui = item.buildUI()
            
            if len(item_ui) > 0:
                components += item_ui
        
        if self.buildResetButton:
            B_UI._buildSeparator()
            btnReset = self.buildButton_Reset(components)
        
        return components
    
    def resetComponentsValues(self, *args) -> any:
        updates = []
        
        bComponents = self.getBComponents()
        for x in bComponents:
            updates.append(x.getUpdate())
        
        if len(updates) == 1:
            updates = updates[0]
        
        return updates
    
    def buildButton_Reset(self, components: list[any]) -> any:
        btnReset = gr.Button(value = f"Reset {self.name}")
        
        btnReset.click(
            fn = self.resetComponentsValues
            , inputs = components
            , outputs = components
        )
        
        return btnReset
    
    def buildUI(self) -> any:
        components = []
        
        with self.buildContainer():
            components = self.buildComponents()
        
        return components
    
    @abstractmethod
    def getDefaultName(self) -> str:
        pass
    
    @abstractmethod
    def buildContainer(self) -> any:
        pass

class B_UI_Preset():
    @staticmethod
    def _fromArgs(mappings: dict[str, list[any]], **kwargs: str):
        return B_UI_Preset(
            mappings = mappings
            , isAdditive = bool(int(kwargs.get("is_additive", 0)))
        )
    
    def __init__(self, mappings: dict[str, list[any]], isAdditive: bool = False):
        self.mappings = mappings
        self.isAdditive = isAdditive
    
    def getPresetValue(self, bComponent: B_UI_Component, componentValue):
        component = bComponent.component
        
        hasPresetValue = component.label in self.mappings
        if not hasPresetValue:
            return bComponent.defaultValue if not self.isAdditive else componentValue
        
        presetValue: any = self.mappings[component.label]
        
        if type(component) is not gr.Dropdown or not component.multiselect:
            presetValue = bComponent.defaultValue if len(presetValue) == 0 else presetValue[0]
        
        return presetValue

class B_Prompt(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def getPositive(self, componentMap: dict[str, B_UI_Component]) -> str:
        pass
    
    @abstractmethod
    def getNegative(self, componentMap: dict[str, B_UI_Component]) -> str:
        pass

class B_Prompt_Simple(B_Prompt):
    @staticmethod
    def _fromArgs(preset: B_UI_Preset, **kwargs: str):
        return B_Prompt_Simple(
            preset = preset
            , promptPositive = kwargs.get("p", "")
            , promptNegative = kwargs.get("n", "")
        )
    
    @staticmethod
    def _createEmpty():
        return B_Prompt_Simple("", "")
    
    def __init__(self, promptPositive: str = "", promptNegative: str = "", preset: B_UI_Preset = None):
        self.positive = promptPositive
        self.negative = promptNegative
        self.preset = preset

        super().__init__()
    
    def getPositive(self, componentMap: dict[str, B_UI_Component]) -> str:
        return self.positive
    
    def getNegative(self, componentMap: dict[str, B_UI_Component]) -> str:
        return self.negative

class B_Prompt_Link_Slider(B_Prompt):
    @staticmethod
    def _fromArgs(**kwargs: str):
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
        
        super().__init__()
    
    def buildPrompt(self, componentMap: dict[str, B_UI_Component]) -> str:
        component = componentMap.get(self.linkedKey)
        
        if component is None:
            print(f"B_Prompt_Link_Slider: Invalid key - '{self.linkedKey}'")
            return ""
        
        if type(component) is not B_UI_Component_Slider:
            print(f"B_Prompt_Link_Slider: Linked entry is not a slider - '{self.linkedKey}'")
            return ""
        
        return buildRangePrompt(self.promptA, self.promptB, component.value)
    
    def getPositive(self, componentMap: dict[str, B_UI_Component]) -> str:
        if self.isNegativePrompt:
            return ""
        
        return self.buildPrompt(componentMap)
    
    def getNegative(self, componentMap: dict[str, B_UI_Component]) -> str:
        if not self.isNegativePrompt:
            return ""
        
        return self.buildPrompt(componentMap)

class B_UI_Component_Textbox(B_UI_Component):
    @staticmethod
    def _fromArgs(name: str, *args, **kwargs: str):
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
    
    def buildComponent(self) -> list[any]:
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
    
    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict[str, B_UI_Component]):
        if not self.isNegativePrompt:
            p.prompt = addPrompt(p.prompt, self.value)
        else:
            p.negative_prompt = addPrompt(p.negative_prompt, self.value)
    
    def getUpdate(self, value = None) -> any:
        return gr.Textbox.update(value = self.handleUpdateValue(value))
    
    def finalizeComponent(self, componentMap: dict):
        pass

class B_UI_Component_Dropdown(B_UI_Component):
    @staticmethod
    def _fromArgs(name: str, *args, **kwargs: str):
        choicesMap: dict[str, B_Prompt] = args[0]

        defaultValues = kwargs.get("v", "")
        if len(defaultValues) > 0:
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
    def _buildColorChoicesMap(postfixPrompt: str = "") -> dict[str, B_Prompt_Simple]:
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
        , choicesMap: dict[str, B_Prompt] = {}
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
    
    def buildChoicesMap(self, choicesMap: dict[str, B_Prompt], insertEmptyChoice: bool) -> dict[str, B_Prompt]:
        choicesMapFinal = choicesMap
        
        if insertEmptyChoice:
            choicesMapFinal = {
                "-": B_Prompt_Simple._createEmpty()
            }
            
            choicesMapFinal.update(choicesMap)
        
        return choicesMapFinal
    
    def buildDefaultValue(self, choicesMap: dict[str, B_Prompt], defaultValues: str | list[str]) -> str | list[str]:
        defaultValue: str | list[str] = None if not self.multiselect else []
        
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
    
    def getChoices(self) -> list[str]:
        choices = list(self.choicesMap)
        
        if not self.sortChoices:
            return choices
        
        choicesSorted: list[str] = []
        
        if choices[0] == "-":
            choicesSorted.append(choices.pop(0))
        
        choices.sort()
        choicesSorted += choices
        
        return choicesSorted
    
    def getBPromptsFromValue(self) -> list[B_Prompt]:
        bPrompts: list[B_Prompt] = []
        
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
    
    def buildComponent(self) -> list[any]:
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
    
    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict[str, B_UI_Component]):
        bPrompts = self.getBPromptsFromValue()
        if len(bPrompts) > 0:
            for bPrompt in bPrompts:
                if bPrompt is not None:
                    p.prompt = addPrompt(p.prompt, bPrompt.getPositive(componentMap))
                    p.negative_prompt = addPrompt(p.negative_prompt, bPrompt.getNegative(componentMap))
    
    def getUpdate(self, value = None) -> any:
        return gr.Dropdown.update(value = self.handleUpdateValue(value))
    
    def finalizeComponent(self, componentMap: dict):
        bComponents: list[B_UI_Component] = componentMap.values()
        components: list[any] = list(map(lambda bComponent: bComponent.component, bComponents))

        anyChoiceMapHasPreset = any(
            map(
                lambda bPrompt: False if type(bPrompt) is not B_Prompt_Simple else bPrompt.preset is not None and len(bPrompt.preset.mappings) > 0
                , self.choicesMap.values()
            )
        )
        if anyChoiceMapHasPreset:
            def getPresetValues(choices: str | list[str], *args):
                if type(choices) is str:
                    choices = [choices]
                
                updatedValues: list[any] = list(args)
                
                if len(choices) > 0:
                    for choice in choices:
                        bPrompt = self.choicesMap[choice]
                        
                        if not issubclass(type(bPrompt), B_Prompt_Simple) or bPrompt.preset == None:
                            continue
                        
                        bPrompt: B_Prompt_Simple = bPrompt
                        
                        i = 0
                        for bComponent in bComponents:
                            updatedValues[i] = bPrompt.preset.getPresetValue(bComponent, updatedValues[i])
                            i += 1
                
                return updatedValues
            
            self.component.select(
                fn = getPresetValues
                , inputs = [self.component] + components
                , outputs = components
            )

class B_UI_Component_Slider(B_UI_Component):
    @staticmethod
    def _fromArgs(name: str, *args, **kwargs: str):
        return B_UI_Component_Slider(
            name = name
            , promptA = kwargs.get("a", "")
            , promptB = kwargs.get("b", "")
            , defaultValue = float(kwargs.get("v", -1))
            , isRequired = bool(int(kwargs.get("is_required", 0)))
            , visible = not bool(int(kwargs.get("hide", 0)))
            , buildButtons = bool(int(kwargs.get("build_buttons", 1)))
            , promptAButton = kwargs.get("a_button", "")
            , promptBButton = kwargs.get("b_button", "")
        )
    
    def __init__(self, name: str = None, promptA: str = "", promptB: str = "", defaultValue: float = None, isRequired: bool = False, visible: bool = True, buildButtons: bool = True, promptAButton: str = "", promptBButton: str = ""):
        self.promptA = promptA.strip()
        self.promptB = promptB.strip()
        self.isRequired = isRequired
        self.buildButtons = buildButtons
        self.promptAButton = promptAButton
        self.promptBButton = promptBButton
        
        self.isNegativePrompt = False
        
        super().__init__(name, defaultValue, visible)
    
    def getMinimum(self) -> float:
        return -1 if not self.isRequired else 0
    
    def getMaximum(self) -> float:
        return 100
    
    def getStep(self) -> float:
        return 1
    
    def buildButton(self, component: any, text: str, value: float) -> any:
        btn = gr.Button(value = text)
        btn.click(
            fn = lambda component: self.getUpdate(value)
            , inputs = component
            , outputs = component
        )
        return btn
    
    def getDefaultName(self) -> str:
        return "Slider"
    
    def buildComponent(self) -> list[any]:
        slider = gr.Slider(
            label = self.name
            , minimum = self.getMinimum()
            , maximum = self.getMaximum()
            , value = self.value
            , step = self.getStep()
            , visible = self.visible
        )
        
        if self.buildButtons:
            with gr.Row():
                self.buildButton(slider, self.promptAButton, 0)
                self.buildButton(slider, self.promptBButton, self.getMaximum())
        
        return [
            slider
        ]
    
    def setValue(self, value: any):
        value = float(value)
        self.value = round(value / self.getMaximum(), 2) if value > -1 else value
    
    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict[str, B_UI_Component]):
        promptToAdd = buildRangePrompt(self.promptA, self.promptB, self.value)
        
        if not self.isNegativePrompt:
            p.prompt = addPrompt(p.prompt, promptToAdd)
        else:
            p.negative_prompt = addPrompt(p.prompt, promptToAdd)
    
    def getUpdate(self, value = None) -> any:
        return gr.Slider.update(value = self.handleUpdateValue(value))
    
    def finalizeComponent(self, componentMap: dict):
        pass

class B_UI_Container_Tab(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str):
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
    
    def buildContainer(self) -> any:
        return gr.Tab(self.name)

class B_UI_Container_Row(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str):
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
    
    def buildContainer(self) -> any:
        return gr.Row(visible = self.visible)

class B_UI_Container_Column(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str):
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
    
    def buildContainer(self) -> any:
        return gr.Column(scale = self.scale, visible = self.visible)

class B_UI_Container_Group(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str):
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
    
    def buildContainer(self) -> any:
        return gr.Group(visible = self.visible)

class B_UI_Container_Accordion(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str):
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
    
    def buildContainer(self) -> any:
        return gr.Accordion(label = self.name, visible = self.visible)

class B_UI_Builder(ABC):
    def __init__(self, name: str, **kwargs: str):
        self.name = name
        self.args = kwargs
    
    @abstractmethod
    def build(self) -> B_UI | B_UI_Preset:
        pass

class B_UI_Builder_WithChildren(B_UI_Builder):
    def __init__(self, name: str, **kwargs: str):
        self.builtChildren: list[B_UI] = []
        
        super().__init__(name, **kwargs)
    
    @abstractmethod
    def build(self) -> B_UI:
        pass

class B_UI_Preset_Builder(B_UI_Builder):
    def __init__(self, name: str, **kwargs: str):
        self.mappings: dict[str, list[str]] = {}
        
        super().__init__(name, **kwargs)
    
    def addMapping(self, name: str, **kwargs: str):
        value = kwargs.get("v", "")
        
        if len(value) > 0:
            value = list(map(lambda v: v.strip(), value.split(",")))
        else:
            value = list[str]([])
        
        self.mappings[name] = value
    
    def build(self) -> B_UI_Preset:
        return None if len(self.mappings) == 0 else B_UI_Preset._fromArgs(self.mappings, **self.args)

class B_UI_Container_Builder(B_UI_Builder_WithChildren):
    def __init__(self, t: type[B_UI_Container], name: str, parent: B_UI_Builder_WithChildren, **kwargs: str):
        self.t = t
        self.parent = parent
        
        super().__init__(name, **kwargs)
    
    def finalizeBuilt(self, built: B_UI):
        if self.parent is not None:
            self.parent.builtChildren.append(built)
        
        return built
    
    def build(self) -> B_UI:
        return self.finalizeBuilt(self.t._fromArgs(self.name, self.builtChildren, **self.args))

class B_UI_Component_Builder(B_UI_Builder):
    def __init__(self, t: type[B_UI_Component], name: str, parent: B_UI_Container_Builder, **kwargs: str):
        self.t = t
        self.parent = parent
        
        super().__init__(name, **kwargs)
    
    def build(self) -> B_UI:
        builtSelf = self.t._fromArgs(self.name, **self.args)
        
        if self.parent is not None:
            self.parent.builtChildren.append(builtSelf)
        
        return builtSelf

class B_UI_Component_Dropdown_Builder(B_UI_Builder):
    def __init__(self, name: str, parent: B_UI_Container_Builder, **kwargs: str):
        self.parent = parent
        
        self.choicesMap: dict[str, B_Prompt] = {}
        
        super().__init__(name, **kwargs)
    
    def addChoice(self, text: str, preset_builder: B_UI_Preset_Builder, **kwargs: str):
        bPrompt: B_Prompt = None
        
        link_type = kwargs.get("link_type", "")
        match link_type:
            case "SLIDER":
                bPrompt = B_Prompt_Link_Slider._fromArgs(**kwargs)
            case _:
                preset = preset_builder.build()
                bPrompt = B_Prompt_Simple._fromArgs(preset, **kwargs)
        
        self.choicesMap[text] = bPrompt
    
    def addChoices(self, **kwargs: str):
        choicesMap: dict[str, B_Prompt] = {}
        
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

class B_UI_Map():
    def __init__(self, path_base: str, file_name_layout: str, file_name_presets: str, tagged_show: bool = True):
        self.layout = self.parseLayout(os.path.join(path_base, file_name_layout), tagged_show)
        self.presets = self.parsePresets(os.path.join(path_base, file_name_presets))
        
        self.componentMap: dict[str, B_UI_Component] = {}
        
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
        dropdown_current_choice: tuple[str, B_UI_Preset_Builder, dict[str, str]] = None
        
        skip = 0
        
        def _build(builder: B_UI_Container_Builder | B_UI_Component_Builder | B_UI_Component_Dropdown_Builder):
            built = builder.build()
            if builder.parent is None:
                layout.append(built)
            return built
        
        def _buildDropdownChoice(dropdown_choice: tuple[str, B_UI_Preset_Builder, dict[str, str]]) -> bool:
            if dropdown_choice is not None:
                l_choice_name, l_choice_preset_builder, l_choice_args = dropdown_choice
                builder_current_dropdown.addChoice(l_choice_name, l_choice_preset_builder, **l_choice_args)
                return len(l_choice_preset_builder.mappings) > 0
            
            return False
        
        with open(file_path_layout) as file_layout:
            for l in file_layout:
                l_type, l_name, l_args = self.readLine(l)
                
                if len(l_type) == 0:
                    continue
                    
                if l_type == ".":
                    break
                
                if l_type == "END":
                    if skip == 0:
                        if dropdown_current_choice is not None:
                            had_mappings = _buildDropdownChoice(dropdown_current_choice)
                            dropdown_current_choice = None
                            if had_mappings:
                                continue

                        if builder_current_dropdown is not None:
                            built = _build(builder_current_dropdown)
                            builder_current_dropdown = None
                            continue
                        
                        if builder_current_container is not None:
                            built = _build(builder_current_container)
                            builder_current_container = builder_current_container.parent
                            continue
                    
                    skip -= 1
                    
                    continue
                
                if l_args.get("x", "") == "1":
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
                        
                        _buildDropdownChoice(dropdown_current_choice)
                        
                        dropdown_current_choice = (l_name, B_UI_Preset_Builder(l_name, **{ "is_additive": "1" }), l_args)
                    
                    case "SET":
                        dropdown_current_choice[1].addMapping(l_name, **l_args)
                    
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
    
    def buildUI(self) -> list[any]:
        B_UI._buildSeparator()
        
        components: list[any] = []
        
        for item in self.layout:
            item_ui = item.buildUI()
            
            if len(item_ui) > 0:
                components += item_ui
        
        self.buildPresetsUI()

        for bComponent in self.componentMap.values():
            bComponent.finalizeComponent(self.componentMap)
        
        return components
    
    def buildPresetsUI(self):
        if len(self.presets) == 0:
            return
        
        bComponents = self.componentMap.values()
        components: list[any] = list(map(lambda bComponent: bComponent.component, bComponents))
        
        presetKeys = self.presets.keys()
        
        def applyPreset(presetKey: str, *args):
            preset = self.presets[presetKey]
            return list(map(preset.getPresetValue, bComponents, args))
        
        B_UI._buildSeparator()
        with gr.Accordion("Presets", open = False):
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
        i = 0
        for k in list(b_layout.componentMap):
            component = b_layout.componentMap[k]
            component.setValue(args[i])
            component.handlePrompt(p, b_layout.componentMap)
            i = i+1
        
        proc = process_images(p)
        
        return proc
