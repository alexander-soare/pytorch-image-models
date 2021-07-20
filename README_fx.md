## Files of interest

**Not for final PR**
- `fx_coverage_results_2.txt` shows the current status of which models can by symbolically traced (and still scriptable)
- `fx_coverage.py` used to generate `fx_coverage_results_2.txt`
- `fx_test.py` if you want to quickly test one model in a similar manner to `fx_coverage.py`
- `fx_features_test.py` - ... if you want a sneak peak.

**Yes for final PR**
- `fx_helpers.py` - Some sneaky workarounds to make code symbolically traceable (see below)
- `fx_features.py` - ... if you want a sneak peak.

## How I made the models symbolically traceable

Here's a list of changes somewhat ranked in order of most impactful to least impactful.

### **Make timm layers into leaf modules**

A leaf module doesn't get traced through, just the reference to it is recorded. This means we can avoid whatever issues come up within. Mostly these have been flow control, and `InplaceAbn`. Check `is_leaf_module` in `MyTracer` of `fx_coverage.py` to see which layers I included.

**Note**: By default, standard torch modules are leaf modules. Maybe there could be consequences of treating timm modules as leaf modules. Not sure.

### **`assert *, *` -> `torch._assert(*, *)`**

Most of the control flow was in assert statements. Luckily we have https://pytorch.org/docs/stable/generated/torch._assert.html

Roughly 10 files needed at least one of these.

### **`* @ *` -> `torch.matmul(*, *)`**

Roughly 15 files, mostly attention related.

### **`* and *` -> `fx_and(*, *)`**

`and` appears in a few of the `torch._asserts`. There we get the symbolic trace control flow error. Custom fx.Tracer class takes this function and wraps it to treat it as a "leaf function".

### **`int` -> `fx_float_to_int`**

Appears in a few places where we are calculating some tensor dim for a reshape/view. Custom fx.Tracer class takes this function and wraps it to treat it as a "leaf function".

### **Don't modify tensor slices in place**

... by using `torch.cat` to reconstruct the full tensor from slices.

In `tnt.py`

```diff
-        patch_embed[:, 1:] = patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))
+        patch_embed = torch.cat(
+            [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
+            dim=1)
```

In `rexnet.py`

```diff
-            x[:, 0:self.in_channels] += shortcut
+            x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
```

## TODO

- Add tracing, forward, backward to pytest. + the jit version
- Could we just treat all submodules as leaf modules and decorate the exceptions rather than the other way around?