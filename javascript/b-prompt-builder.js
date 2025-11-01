onUiLoaded(() => {
	const elem_template = document.getElementById('b-ui-i1');
	if (!elem_template) return;
	window.addEventListener('click', e => {
		if (!e.target.matches('.b-ui-select-choice')) return;
		e.target.parentElement.insertAdjacentElement('afterend', elem_template);
	})
});
